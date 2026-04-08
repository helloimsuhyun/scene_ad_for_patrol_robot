import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';
import 'package:http/http.dart' as http;

// 서버 주소 (중앙 집중 관리 권장)
const String _imageUrlBase = 'http://localhost:8000/images/';

String formatEventTime(String isoString) {
  try {
    final dt = DateTime.parse(isoString).toLocal();
    return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}';
  } catch (_) {
    return '--:--:--';
  }
}

void showEventDetailDialog(BuildContext context, WidgetRef ref, Event event) {
  // 선택된 이벤트를 상태로 저장
  ref.read(selectedEventProvider.notifier).state = event;

  showDialog(
    context: context,
    builder: (context) {
      return Dialog(
        backgroundColor: const Color(0xFF1C1E2B),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Container(
          width: 500,
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    event.anomalyFlag == 1 ? '🚨 이상 감지 내역' : 'ℹ️ 시스템 알림',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.network(
                  event.frames.isNotEmpty
                      ? '$_imageUrlBase${event.frames.first.imagePath.replaceFirst("recv/", "")}'
                      : 'https://via.placeholder.com/500x300.png?text=No+Image',
                  width: double.infinity,
                  height: 300,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      width: double.infinity,
                      height: 300,
                      color: const Color(0xFF26293A),
                      child: const Center(
                        child: Text(
                          '이미지를 불러올 수 없습니다.\n서버 연결 상태를 확인하세요.',
                          textAlign: TextAlign.center,
                          style: TextStyle(color: Color(0xFF7A7F96)),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),
              Text(
                '요약: ${event.summaryText ?? "이벤트 발생"}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (event.adminLabel != null) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Icon(Icons.check_circle, color: Color(0xFF4ADE80), size: 16),
                    const SizedBox(width: 6),
                    Text(
                      '관리자 확인 완료 (${event.adminLabel})',
                      style: const TextStyle(
                        color: Color(0xFF4ADE80),
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ],
              const SizedBox(height: 20),
              const Divider(color: Colors.white12, height: 1),
              const SizedBox(height: 16),
              _DetailRow(
                label: '발생 시간',
                value: formatEventTime(event.capturedAt),
              ),
              const SizedBox(height: 12),
              _DetailRow(label: '장소 ID', value: event.placeId),
              if (event.anomalyScore != null) ...[
                const SizedBox(height: 12),
                _DetailRow(
                  label: '위험도 (Confidence)',
                  value: '${(event.anomalyScore! * 100).toStringAsFixed(1)}%',
                  valueColor: const Color(0xFFEF4444),
                ),
              ],
              const SizedBox(height: 16),
              const Divider(color: Colors.white12, height: 1),
              const SizedBox(height: 24),
              //---------- 시스템 오탐지 피드백 버튼 영역 ----------
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(
                          0xFF22C55E,
                        ).withOpacity(0.15),
                        foregroundColor: const Color(0xFF22C55E),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      icon: const Icon(Icons.check_circle_outline, size: 18),
                      label: const Text('정상 (오탐지 - 뱅크 추가)'),
                      onPressed: () async {
                        try {
                          Navigator.of(context).pop();
                          final response = await http.post(
                            Uri.parse('http://127.0.0.1:8000/move_event'),
                            headers: {'Content-Type': 'application/json'},
                            body:
                                '{"event_id": "${event.eventId}", "move": true}',
                          );
                          if (response.statusCode == 200) {
                            // 리스트 즉시 갱신
                            ref.invalidate(eventListProvider);
                          } else {
                            debugPrint(
                              'Failed to move event: ${response.body}',
                            );
                          }
                        } catch (e) {
                          debugPrint('Error moving event: $e');
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFFEF4444),
                        side: const BorderSide(color: Color(0xFFEF4444)),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      icon: const Icon(Icons.warning_amber_rounded, size: 18),
                      label: const Text('파손/침입'),
                      onPressed: () async {
                        Navigator.of(context).pop();
                        try {
                          await http.patch(
                            Uri.parse('http://127.0.0.1:8000/events/${event.eventId}/label'),
                            headers: {'Content-Type': 'application/json'},
                            body: '{"admin_label": "파손/침입"}',
                          );
                          // 리스트 즉시 갱신
                          ref.invalidate(eventListProvider);
                        } catch (e) {
                          debugPrint('Label error: $e');
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFFF97316),
                        side: const BorderSide(color: Color(0xFFF97316)),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      icon: const Icon(Icons.error_outline, size: 18),
                      label: const Text('기타 이상'),
                      onPressed: () async {
                        Navigator.of(context).pop();
                        try {
                          await http.patch(
                            Uri.parse('http://127.0.0.1:8000/events/${event.eventId}/label'),
                            headers: {'Content-Type': 'application/json'},
                            body: '{"admin_label": "기타 이상"}',
                          );
                          // 리스트 즉시 갱신
                          ref.invalidate(eventListProvider);
                        } catch (e) {
                          debugPrint('Label error: $e');
                        }
                      },
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      );
    },
  );
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;

  const _DetailRow({required this.label, required this.value, this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 13),
          ),
          Text(
            value,
            style: TextStyle(
              color: valueColor ?? Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
