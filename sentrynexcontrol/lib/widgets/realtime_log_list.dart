import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../providers/audio_provider.dart';
import '../providers/yolo_provider.dart';
import '../models/event_model.dart';
import '../models/audio_event_model.dart';
import '../models/yolo_event_model.dart';
import 'event_detail_dialog.dart';

class RealtimeLogList extends ConsumerWidget {
  const RealtimeLogList({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final visionEvents = ref.watch(eventListProvider);
    final audioEvents = ref.watch(audioEventListProvider);
    final yoloEvents = ref.watch(yoloEventsProvider);

    // 각 이벤트 타입별로 확인되지 않은(unchecked) 항목만 필터링
    final List<dynamic> allUncheckedEvents = [];

    allUncheckedEvents.addAll(visionEvents.where(
        (e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));
    
    allUncheckedEvents.addAll(audioEvents.where(
        (e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));
        
    allUncheckedEvents.addAll(yoloEvents.where(
        (e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));

    // 시간 역순(최신순) 정렬
    allUncheckedEvents.sort((a, b) {
      final tA = DateTime.parse(a.timestamp).toLocal();
      final tB = DateTime.parse(b.timestamp).toLocal();
      return tB.compareTo(tA);
    });

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: const [
            Icon(Icons.warning_amber_outlined, size: 14, color: Color(0xFFB5BAD3)),
            SizedBox(width: 6),
            Text(
              '실시간 미확인 로그',
              style: TextStyle(
                color: Color(0xFFB5BAD3),
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Expanded(
          child: allUncheckedEvents.isEmpty
              ? const Center(
                  child: Text(
                    '미확인 경보가 없습니다.',
                    style: TextStyle(color: Color(0xFF4A4E63), fontSize: 12),
                  ),
                )
              : ListView.separated(
                  padding: EdgeInsets.zero,
                  itemBuilder: (context, index) {
                    final event = allUncheckedEvents[index];
                    return _RealtimeAlertTile(event: event);
                  },
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemCount: allUncheckedEvents.length,
                ),
        ),
      ],
    );
  }
}

class _RealtimeAlertTile extends ConsumerWidget {
  final dynamic event;

  const _RealtimeAlertTile({required this.event});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    Color badgeColor;
    String badgeLabel;
    String summaryStr;
    String placeStr;
    
    if (event is Event) {
      badgeColor = event.anomalyFlag == 1 ? const Color(0xFFEF4444) : const Color(0xFF38BDF8);
      badgeLabel = event.anomalyFlag == 1 ? 'VISION ALARM' : 'VISION INFO';
      summaryStr = event.summaryText ?? '비전 이벤트';
      placeStr = event.placeId;
    } else if (event is AudioEvent) {
      badgeColor = const Color(0xFFBA68C8);
      badgeLabel = 'AUDIO ALARM';
      summaryStr = event.modelLabel ?? '오디오 이벤트';
      placeStr = '오디오 감시구역'; // 오디오는 event.placeId 가 없을 수 있음
    } else if (event is YoloEvent) {
      badgeColor = const Color(0xFF4ADE80);
      badgeLabel = 'YOLO ALARM';
      summaryStr = '${event.eventType ?? "인물 감지"} (${event.personCount}명)';
      placeStr = event.sourceRegionName ?? 'YOLO 감시구역';
    } else {
      return const SizedBox.shrink();
    }

    return GestureDetector(
      onDoubleTap: () {
        if (event is Event) {
          showEventDetailDialog(context, ref, event);
        } else if (event is AudioEvent) {
          showAudioEventDetailDialog(context, ref, event);
        } else if (event is YoloEvent) {
          showYoloEventDetailDialog(context, ref, event);
        }
      },
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFF1C1E2B), // 사이드바 보다는 약간 밝은 톤
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0xFF2E3244)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: badgeColor.withOpacity(0.14),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    badgeLabel,
                    style: TextStyle(color: badgeColor, fontSize: 9, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              summaryStr,
              style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600, height: 1.3),
            ),
            const SizedBox(height: 6),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.access_time, size: 10, color: Color(0xFF757B92)),
                    const SizedBox(width: 4),
                    Text(
                      formatEventTime(event.timestamp),
                      style: const TextStyle(color: Color(0xFF757B92), fontSize: 9),
                    ),
                  ],
                ),
                Row(
                  children: [
                    const Icon(Icons.location_on_outlined, size: 10, color: Color(0xFF757B92)),
                    const SizedBox(width: 4),
                    Text(
                      placeStr,
                      style: const TextStyle(color: Color(0xFF757B92), fontSize: 9),
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
