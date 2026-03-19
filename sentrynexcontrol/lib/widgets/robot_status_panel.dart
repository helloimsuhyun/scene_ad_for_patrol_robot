import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// 수동 캡처 대기를 위한 로딩 상태
final _isCapturingProvider = StateProvider<bool>((ref) => false);

// 수동 캡처 테스트를 위한 현재 라벨 상태 (z 명령어 토글용)
final _queryLabelProvider = StateProvider<String>((ref) => 'normal');

class RobotStatusPanel extends ConsumerStatefulWidget {
  const RobotStatusPanel({super.key});

  @override
  ConsumerState<RobotStatusPanel> createState() => _RobotStatusPanelState();
}

class _RobotStatusPanelState extends ConsumerState<RobotStatusPanel> {
  bool isCliButtonsEnabled = false;

  Future<void> _triggerCapture(
    BuildContext context,
    WidgetRef ref,
    String endpoint,
  ) async {
    final loadingNotifier = ref.read(_isCapturingProvider.notifier);
    loadingNotifier.state = true;
    try {
      final response = await http.post(
        Uri.parse('http://192.168.0.88:8090/patrol/$endpoint'),
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['image_b64'] != null && context.mounted) {
          final bytes = base64Decode(data['image_b64']);
          showDialog(
            context: context,
            builder: (ctx) => Dialog(
              backgroundColor: const Color(0xFF1C1E2B),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      '📸 캡처 완료!',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(
                        bytes,
                        height: 300,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF1F8CEB),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 12),
                        ),
                        onPressed: () => Navigator.of(ctx).pop(),
                        child: const Text(
                          '확인',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
      } else {
        if (context.mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(const SnackBar(content: Text('캡처 명령 실패')));
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('에러 발생: $e')));
      }
    } finally {
      loadingNotifier.state = false;
    }
  }

  Future<void> _toggleQueryLabel(WidgetRef ref) async {
    final current = ref.read(_queryLabelProvider);
    final next = current == 'normal' ? 'abnormal' : 'normal';
    try {
      await http.post(
        Uri.parse('http://192.168.0.88:8090/patrol/query_gt'),
        headers: {'Content-Type': 'application/json'},
        body: '{"label": "$next"}',
      );
      ref.read(_queryLabelProvider.notifier).state = next;
    } catch (e) {
      debugPrint('Error toggling label: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12), // 패널 안쪽 여백 줄임 (로그창 공간 확보)
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Icon(
                Icons.smart_toy_outlined,
                size: 18,
                color: Color(0xFFB5BAD3),
              ),
              const SizedBox(width: 6),
              const Text(
                'Robot Status',
                style: TextStyle(
                  color: Color(0xFFB5BAD3),
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              //---------- 우상단 소형 배터리 인디케이터 ----------
              Row(
                children: [
                  Container(
                    width: 60, // 배터리 막대 가로 늘림
                    height: 14, // 세로 늘림
                    decoration: BoxDecoration(
                      color: const Color(0xFF26293A),
                      borderRadius: BorderRadius.circular(5),
                    ),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Container(
                        width: 60 * 0.72,
                        decoration: BoxDecoration(
                          color: const Color(0xFF7F7CFF), // 포인트 컬러 연보라색
                          borderRadius: BorderRadius.circular(5),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    '72%',
                    style: TextStyle(
                      color: Color(0xFF7F7CFF), // 연보라색 텍스트
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              _RobotMetric(
                label: '연결 상태',
                value: '온라인',
                color: Color(0xFF4ADE80),
              ),
              _RobotMetric(
                label: '현재 속도',
                value: '1.2 m/s',
                color: Color(0xFFB5BAD3),
              ),
              _RobotMetric(
                label: '작업 모드',
                value: '순찰 중',
                color: Color(0xFFB5BAD3),
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Divider(height: 1, color: Color(0xFF2D3041)),
          const SizedBox(height: 10),
          //---------- 명령어 컨트롤러 헤더 및 스위치 ----------
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '명령어 컨트롤러',
                style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
              ),
              Switch(
                value: isCliButtonsEnabled,
                inactiveThumbColor: const Color(0xFF7F7CFF),
                activeColor: const Color(0xFF7F7CFF),
                onChanged: (value) {
                  setState(() {
                    isCliButtonsEnabled = value;
                  });
                },
              ),
            ],
          ),

          if (isCliButtonsEnabled) ...[
            const SizedBox(height: 6),
            Builder(
              builder: (context) {
                final queryLabel = ref.watch(_queryLabelProvider);
                final isCapturing = ref.watch(_isCapturingProvider);

                return Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        Expanded(
                          child: TextButton.icon(
                            onPressed: () => _toggleQueryLabel(ref),
                            icon: Icon(
                              queryLabel == 'normal'
                                  ? Icons.shield_outlined
                                  : Icons.warning_amber_rounded,
                              color: queryLabel == 'normal'
                                  ? Colors.greenAccent
                                  : Colors.redAccent,
                              size: 14,
                            ),
                            label: Text(
                              '라벨: $queryLabel',
                              style: TextStyle(
                                color: queryLabel == 'normal'
                                    ? Colors.greenAccent
                                    : Colors.redAccent,
                                fontSize: 11,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: isCapturing
                                ? null
                                : () =>
                                      _triggerCapture(context, ref, 'capture'),
                            icon: isCapturing
                                ? const SizedBox(
                                    width: 14,
                                    height: 14,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white54,
                                    ),
                                  )
                                : const Icon(
                                    Icons.camera_alt_outlined,
                                    size: 14,
                                  ),
                            label: const Text(
                              '현재캡처',
                              style: TextStyle(fontSize: 11),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF26293A),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 8),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: isCapturing
                                ? null
                                : () => _triggerCapture(
                                    context,
                                    ref,
                                    'place_and_capture',
                                  ),
                            icon: isCapturing
                                ? const SizedBox(
                                    width: 14,
                                    height: 14,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white54,
                                    ),
                                  )
                                : const Icon(
                                    Icons.location_on_outlined,
                                    size: 14,
                                  ),
                            label: const Text(
                              '이동+캡처',
                              style: TextStyle(fontSize: 11),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF1F8CEB),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 8),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                );
              },
            ),
          ],
        ],
      ),
    );
  }
}

class _RobotMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;

  const _RobotMetric({required this.label, required this.value, this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color ?? Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}
