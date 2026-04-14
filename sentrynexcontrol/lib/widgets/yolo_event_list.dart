import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/yolo_provider.dart';
import '../models/yolo_event_model.dart';

class YoloEventList extends ConsumerWidget {
  final bool showOnlyUnchecked;
  const YoloEventList({super.key, this.showOnlyUnchecked = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final allEvents = ref.watch(yoloEventsProvider);
    final events = List<YoloEvent>.from(showOnlyUnchecked
        ? allEvents.where((e) {
            final bool isChecked = (e.adminChecked == 1) || (e.adminLabel != null && e.adminLabel!.isNotEmpty);
            return !isChecked;
          })
        : allEvents);

    events.sort((a, b) {
      bool aChecked = (a.adminChecked == 1) || (a.adminLabel != null && a.adminLabel!.isNotEmpty);
      bool bChecked = (b.adminChecked == 1) || (b.adminLabel != null && b.adminLabel!.isNotEmpty);
      
      if (aChecked != bChecked) {
        return aChecked ? 1 : -1;
      }
      
      final tA = DateTime.parse(a.timestamp).toLocal();
      final tB = DateTime.parse(b.timestamp).toLocal();
      return tB.compareTo(tA);
    });

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
            children: const [
              Icon(Icons.person_pin_circle_outlined, size: 18, color: Color(0xFFB5BAD3)),
              SizedBox(width: 6),
              Text(
                '사람 감지 (YOLO) 이벤트 내역',
                style: TextStyle(
                  color: Color(0xFFB5BAD3),
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: events.isEmpty
                ? const Center(
                    child: Text(
                      '새로운 YOLO 이벤트가 없습니다.',
                      style: TextStyle(color: Color(0xFF4A4E63)),
                    ),
                  )
                : ListView.separated(
                    itemBuilder: (context, index) {
                      final event = events[index];
                      return _YoloTile(event: event, showOnlyUnchecked: showOnlyUnchecked);
                    },
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemCount: events.length,
                  ),
          ),
        ],
      ),
    );
  }
}

class _YoloTile extends ConsumerWidget {
  final YoloEvent event;
  final bool showOnlyUnchecked;

  const _YoloTile({required this.event, this.showOnlyUnchecked = false});

  String formatLocalTime(String ts) {
    try {
      final dt = DateTime.parse(ts).toLocal();
      final timeStr = '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}';
      if (showOnlyUnchecked) return timeStr;
      
      final dateStr = '${dt.year.toString().substring(2)}/${dt.month.toString().padLeft(2, '0')}/${dt.day.toString().padLeft(2, '0')}';
      return '$dateStr $timeStr';
    } catch (_) {
      return '--:--:--';
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool isDwelling = event.eventType == 'person_dwelling';
    final bool isChecked = event.adminChecked == 1 || event.adminLabel != null;

    String badgeLabel = isDwelling ? 'DWELLING' : 'PRESENT';
    Color badgeColor = isDwelling ? const Color(0xFFEF4444) : const Color(0xFFEAB308);

    if (isChecked) {
      badgeLabel = 'CHECKED';
      badgeColor = const Color(0xFF7A7F96);
    }

    return Opacity(
      opacity: isChecked ? 0.7 : 1.0,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF11131C),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isChecked
                ? const Color(0xFF2D3041)
                : badgeColor.withOpacity(0.5),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: badgeColor.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    badgeLabel,
                    style: TextStyle(
                      color: badgeColor,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                if (isChecked)
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.check_circle, color: Color(0xFF4ADE80), size: 13),
                      const SizedBox(width: 4),
                      Text(
                        event.adminLabel ?? '확인됨',
                        style: const TextStyle(
                          color: Color(0xFF4ADE80),
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              '감지 인원: ${event.personCount}명 ${event.sourceRegionName != null ? '(${event.sourceRegionName})' : ''}',
              style: TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: isChecked ? FontWeight.normal : FontWeight.w600,
              ),
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.access_time, size: 10, color: Color(0xFF757B92)),
                    const SizedBox(width: 4),
                    Text(
                      formatLocalTime(event.timestamp),
                      style: const TextStyle(
                        color: Color(0xFF757B92),
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
                if (event.imageUrl != null)
                  const Row(
                    children: [
                      Icon(Icons.image_outlined, size: 10, color: Color(0xFF757B92)),
                      SizedBox(width: 4),
                      Text(
                        '이미지 첨부됨',
                        style: TextStyle(color: Color(0xFF757B92), fontSize: 11),
                      ),
                    ],
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
