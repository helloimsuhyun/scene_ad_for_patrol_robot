import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';
import 'event_detail_dialog.dart';

class AlertList extends ConsumerWidget {
  const AlertList({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Riverpod 상태에서 실시간 이벤트 목록 가져오기
    final alerts = ref.watch(eventListProvider);

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: const [
                  Icon(
                    Icons.warning_amber_outlined,
                    size: 18,
                    color: Color(0xFFB5BAD3),
                  ),
                  SizedBox(width: 6),
                  Text(
                    '실시간 로그',
                    style: TextStyle(
                      color: Color(0xFFB5BAD3),
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
              // 더미 데이터 테스트용 버튼
              IconButton(
                icon: const Icon(Icons.add_circle_outline, color: Color(0xFF26293A), size: 18),
                onPressed: () {
                  ref.read(eventListProvider.notifier).generateMockEvent();
                },
                tooltip: '테스트 알림 추가',
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            '최근 경보를 확인하려면 더블클릭하세요.',
            style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 12),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: alerts.isEmpty
                ? const Center(
                    child: Text(
                      '새로운 이벤트가 없습니다.',
                      style: TextStyle(color: Color(0xFF4A4E63)),
                    ),
                  )
                : ListView.separated(
                    itemBuilder: (context, index) {
                      final event = alerts[index];
                      return _AlertTile(event: event);
                    },
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemCount: alerts.length,
                  ),
          ),
        ],
      ),
    );
  }
}

class _AlertTile extends ConsumerWidget {
  final Event event;

  const _AlertTile({required this.event});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool isAnomaly = event.anomalyFlag == 1;
    final Color color = isAnomaly ? const Color(0xFFEF4444) : const Color(0xFF38BDF8);
    final String level = isAnomaly ? 'WARNING' : 'INFO';

    return GestureDetector(
      onDoubleTap: () => showEventDetailDialog(context, ref, event),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF11131C),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isAnomaly ? const Color(0x33EF4444) : const Color(0xFF2E3244),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.14),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                level,
                style: TextStyle(
                  color: color,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.6,
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    event.summaryText ?? '알 수 없는 이벤트',
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        formatEventTime(event.capturedAt),
                        style: const TextStyle(
                          color: Color(0xFF757B92),
                          fontSize: 11,
                        ),
                      ),
                      Text(
                        event.placeId,
                        style: const TextStyle(
                          color: Color(0xFF757B92),
                          fontSize: 11,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
