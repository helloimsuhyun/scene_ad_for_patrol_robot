import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';
import 'event_detail_dialog.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// 수동 캡처 대기를 위한 로딩 상태
final _isCapturingProvider = StateProvider<bool>((ref) => false);

// 수동 캡처 테스트를 위한 현재 라벨 상태 (z 명령어 토글용)
final _queryLabelProvider = StateProvider<String>((ref) => 'normal');

// 맵 상에서 클릭된 이벤트를 추적하기 위한 로컬 상태
final _selectedMapEventProvider = StateProvider<String?>((ref) => null);

class DataCenterMap extends ConsumerWidget {
  const DataCenterMap({super.key});

  Future<void> _triggerCapture(BuildContext context, WidgetRef ref, String endpoint) async {
    final loadingNotifier = ref.read(_isCapturingProvider.notifier);
    loadingNotifier.state = true;
    try {
      final response = await http.post(Uri.parse('http://192.168.0.88:8090/patrol/$endpoint'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['image_b64'] != null && context.mounted) {
          final bytes = base64Decode(data['image_b64']);
          showDialog(
            context: context,
            builder: (ctx) => Dialog(
              backgroundColor: const Color(0xFF1C1E2B),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text('📸 캡처 완료!', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 16),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(bytes, height: 300, fit: BoxFit.cover),
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
                        child: const Text('확인', style: TextStyle(fontWeight: FontWeight.bold)),
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
          ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('캡처 명령 실패')));
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('에러 발생: $e')));
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
  Widget build(BuildContext context, WidgetRef ref) {
    final alerts = ref.watch(eventListProvider);
    final queryLabel = ref.watch(_queryLabelProvider);

    //---------- 맵에 표시할 고유한 장소별 최신 이벤트 필터링 ----------
    final Map<String, Event> latestEventsByPlace = {};
    for (var event in alerts) {
      if (!latestEventsByPlace.containsKey(event.placeId)) {
        latestEventsByPlace[event.placeId] = event;
      }
    }

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 맵 본체 프레임 영역 ----------
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xFF10121A),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  //---------- 맵 배경 격자(Grid) 무늬 레이어 ----------
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _MapGridPainter(),
                    ),
                  ),
                  //---------- 각 장소별 최신 이벤트 마커 표시 ----------
                  ...latestEventsByPlace.values.map((event) => _EventMarker(event: event)),
                  
                  //---------- 수동 제어 플로팅 패널 (로봇 캡처 및 테스트용) ----------
                  if (false) Positioned(
                    top: 16,
                    right: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1C1E2B).withOpacity(0.85),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFF2D3041)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          TextButton.icon(
                            onPressed: () => _toggleQueryLabel(ref),
                            icon: Icon(
                              queryLabel == 'normal' ? Icons.shield_outlined : Icons.warning_amber_rounded,
                              color: queryLabel == 'normal' ? Colors.greenAccent : Colors.redAccent,
                              size: 16,
                            ),
                            label: Text(
                              '라벨: $queryLabel (z)',
                              style: TextStyle(color: queryLabel == 'normal' ? Colors.greenAccent : Colors.redAccent, fontSize: 12),
                            ),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton.icon(
                            onPressed: ref.watch(_isCapturingProvider) ? null : () => _triggerCapture(context, ref, 'capture'),
                            icon: ref.watch(_isCapturingProvider) 
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white54))
                              : const Icon(Icons.camera_alt_outlined, size: 16),
                            label: const Text('현재캡처 (c)', style: TextStyle(fontSize: 12)),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF26293A),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                            ),
                          ),
                          const SizedBox(width: 8),
                          ElevatedButton.icon(
                            onPressed: ref.watch(_isCapturingProvider) ? null : () => _triggerCapture(context, ref, 'place_and_capture'),
                            icon: ref.watch(_isCapturingProvider) 
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white54))
                              : const Icon(Icons.location_on_outlined, size: 16),
                            label: const Text('이동+캡처 (v)', style: TextStyle(fontSize: 12)),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF1F8CEB),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _EventMarker extends ConsumerWidget {
  final Event event;
  const _EventMarker({required this.event});

  //---------- 특정 장소 ID별로 고정된 지도 상의 위치 (원하는 위치로 수정 가능) ----------
  static const Map<String, Alignment> _fixedPositions = {
    '00': Alignment(-0.85, -0.65), // 00번 장소
    '01': Alignment(0.75, 0.45),   // 01번 장소
    '02': Alignment(0.0, 0.85),    // 02번 장소
  };

  Alignment _getAlignmentForPlace(String placeId) {
    // 1. 먼저 고정된 위치값이 있는지 확인
    if (_fixedPositions.containsKey(placeId)) {
      return _fixedPositions[placeId]!;
    }

    // 2. 등록되지 않은 ID라면 장소 ID에 기반한 간단한 해시로 맵 상의 고정 좌표 생성
    final hash = placeId.hashCode;
    final x = (((hash % 100) / 50.0) - 1.0) * 0.9;
    final y = ((((hash ~/ 100) % 100) / 50.0) - 1.0) * 0.9;
    return Alignment(x, y);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAnomaly = event.anomalyFlag == 1;
    final selectedId = ref.watch(_selectedMapEventProvider);
    final isSelected = selectedId == event.eventId;

    //---------- 히트 테스트(터치 인식) 영역 문제 해결 ----------
    // 히트 테스트 영역을 확보하면서 마커 위치가 튀지 않도록 전체 크기를 항상 일정하게(220x320) 고정합니다.
    // (Align 위젯은 자식의 가로/세로 크기가 변하면 중심 좌표를 다시 계산하기 때문에 위치가 튀는 버그가 발생함)
    return Align(
      alignment: _getAlignmentForPlace(event.placeId),
      child: SizedBox(
        width: 220, 
        height: 320,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center, // 마커를 박스 정중앙(중심점)에 배치
          children: [
            // 말풍선 (선택 시 마커 상단에 표시)
            if (isSelected)
              Positioned(
                bottom: 160 + 15, // 박스 중앙(160) + 마커 높이 절반 이상 여유
                child: GestureDetector(
                  onTap: () => showEventDetailDialog(context, ref, event),
                  child: Container(
                    width: 210,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1E2B),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isAnomaly ? const Color(0xFFEF4444) : const Color(0xFF38BDF8),
                        width: 1.5,
                      ),
                      boxShadow: const [
                        BoxShadow(color: Colors.black54, blurRadius: 15, offset: Offset(0, 8)),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Text(
                                event.summaryText ?? '이벤트 발생',
                                style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            const Icon(Icons.open_in_new, color: Colors.white54, size: 14),
                          ],
                        ),
                        const SizedBox(height: 8),
                        // 이미지 영역
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Container(
                            height: 110,
                            width: double.infinity,
                            decoration: const BoxDecoration(color: Color(0xFF11121A)),
                            child: event.frames.isNotEmpty
                                ? Image.network(
                                    'http://localhost:8000/images/${event.frames.first.imagePath.replaceFirst("recv/", "")}',
                                    fit: BoxFit.cover,
                                    errorBuilder: (_, __, ___) => const Icon(Icons.broken_image, color: Colors.grey, size: 24),
                                  )
                                : const Icon(Icons.image_not_supported, color: Colors.grey, size: 24),
                          ),
                        ),
                        const SizedBox(height: 6),
                        const Center(
                          child: Text(
                            '클릭하여 상세 정보 확인',
                            style: TextStyle(color: Colors.white38, fontSize: 10, fontStyle: FontStyle.italic),
                          ),
                        ),
                        const SizedBox(height: 6),
                        // 모드 변경 (Control Provider 연동 직접 팝업)
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton(
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 4),
                              side: const BorderSide(color: Color(0xFF393C4B)),
                            ),
                            onPressed: () async {
                              // 모드 강제 전환 로직 (모드 팝업 메뉴)
                              showMenu(
                                context: context,
                                position: const RelativeRect.fromLTRB(100, 100, 0, 0),
                                color: const Color(0xFF1C1E2B),
                                items: const [
                                  PopupMenuItem(value: 'idle', child: Text('idle', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'bank', child: Text('bank', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'th_calib', child: Text('th_calib', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'query', child: Text('query', style: TextStyle(color: Colors.white))),
                                ],
                              ).then((value) {
                                if (value != null) {
                                  http.post(
                                    Uri.parse('http://127.0.0.1:8000/places/${event.placeId}/config'),
                                    body: {'mode': value},
                                  ).then((_) {
                                    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('${event.placeId} 구역 모드 변경: $value')));
                                  });
                                }
                              });
                            },
                            child: const Text('운영 모드 변경 (m)', style: TextStyle(fontSize: 10, color: Color(0xFFB5BAD3))),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            
            // 실제 표시되는 마커 본체 (중앙 유지)
            GestureDetector(
              onTap: () {
                final notifier = ref.read(_selectedMapEventProvider.notifier);
                notifier.state = (notifier.state == event.eventId) ? null : event.eventId;
              },
              child: isAnomaly
                  ? const _PulsingDot(color: Color(0xFFFF4B5C), size: 24) // 비정상일 경우 펄스 마커
                  : Container(
                      width: 16,
                      height: 16,
                      decoration: BoxDecoration(
                        color: const Color(0xFF38BDF8),
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 2),
                        boxShadow: const [
                          BoxShadow(
                            color: Color(0x6638BDF8),
                            blurRadius: 10,
                            spreadRadius: 2,
                          )
                        ],
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MapGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF26293A)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    const double step = 40;

    for (double x = 0; x <= size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y <= size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// 비정상 마커용 펄스(Ripple) 애니메이션 마커
class _PulsingDot extends StatefulWidget {
  final Color color;
  final double size;
  const _PulsingDot({required this.color, required this.size});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: Stack(
        alignment: Alignment.center,
        clipBehavior: Clip.none,
        children: [
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return Opacity(
                opacity: 1.0 - _controller.value,
                child: Container(
                  width: widget.size + (widget.size * 2 * _controller.value),
                  height: widget.size + (widget.size * 2 * _controller.value),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: widget.color.withOpacity(0.5),
                  ),
                ),
              );
            },
          ),
          Container(
            width: widget.size,
            height: widget.size,
            decoration: BoxDecoration(shape: BoxShape.circle, color: widget.color),
            child: const Icon(Icons.warning_rounded, size: 16, color: Colors.white),
          ),
        ],
      ),
    );
  }
}
