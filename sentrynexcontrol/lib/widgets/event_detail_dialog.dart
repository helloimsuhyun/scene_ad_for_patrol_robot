import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';

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
                style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFF11121A),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  children: [
                    _DetailRow(label: '발생 시간', value: formatEventTime(event.capturedAt)),
                    const Divider(color: Colors.white10),
                    _DetailRow(label: '장소 ID', value: event.placeId),
                    if (event.anomalyScore != null) ...[
                      const Divider(color: Colors.white10),
                      _DetailRow(
                        label: '위험도', 
                        value: '${(event.anomalyScore! * 100).toStringAsFixed(1)}%',
                        valueColor: const Color(0xFFEF4444),
                      ),
                    ],
                  ],
                ),
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
          Text(label, style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 13)),
          Text(value, style: TextStyle(color: valueColor ?? Colors.white, fontSize: 13, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }
}
