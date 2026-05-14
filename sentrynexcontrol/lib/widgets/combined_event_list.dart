import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../providers/audio_provider.dart';
import '../providers/yolo_provider.dart';
import '../providers/auth_event_provider.dart';
import '../models/event_model.dart';
import '../models/audio_event_model.dart';
import '../models/yolo_event_model.dart';
import '../models/auth_event_model.dart';
import 'event_detail_dialog.dart';
import 'auth_event_detail_dialog.dart';

class CombinedEventList extends ConsumerWidget {
  final bool showOnlyUnchecked;
  final String? title;

  const CombinedEventList({
    super.key,
    this.showOnlyUnchecked = false,
    this.title,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final visionEvents = ref.watch(eventListProvider);
    final audioEvents = ref.watch(audioEventListProvider);
    final yoloEvents = ref.watch(yoloEventsProvider);
    final authEvents = ref.watch(authEventListProvider);

    final List<dynamic> allEvents = [];

    if (showOnlyUnchecked) {
      allEvents.addAll(visionEvents.where((e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));
      allEvents.addAll(audioEvents.where((e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));
      allEvents.addAll(yoloEvents.where((e) => (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty))));
      // 2차인증 미확인 경보: success/fail/timeout 중 아직 admin이 처리 안 한 것
      allEvents.addAll(authEvents.where((e) => e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty)));
    } else {
      allEvents.addAll(visionEvents);
      allEvents.addAll(audioEvents);
      allEvents.addAll(yoloEvents);
      allEvents.addAll(authEvents);
    }

    // 미확인 먼저, 동일 그룹 내에선 최신순(시간 역순) 정렬
    allEvents.sort((a, b) {
      bool aChecked = (a.adminChecked == 1) || (a.adminLabel != null && a.adminLabel!.isNotEmpty);
      bool bChecked = (b.adminChecked == 1) || (b.adminLabel != null && b.adminLabel!.isNotEmpty);
      
      if (aChecked != bChecked) {
        return aChecked ? 1 : -1;
      }
      
      // AuthEvent는 timestamp, Event는 capturedAt, 나머지는 timestamp 사용
      String tsA = a is Event ? a.capturedAt : a.timestamp;
      String tsB = b is Event ? b.capturedAt : b.timestamp;
      final tA = DateTime.tryParse(tsA)?.toLocal() ?? DateTime(2000);
      final tB = DateTime.tryParse(tsB)?.toLocal() ?? DateTime(2000);
      return tB.compareTo(tA);
    });

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (title != null) ...[
          Row(
            children: [
              const Icon(Icons.warning_amber_outlined, size: 14, color: Color(0xFFB5BAD3)),
              const SizedBox(width: 6),
              Text(
                title!,
                style: const TextStyle(
                  color: Color(0xFFB5BAD3),
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
        ],
        Expanded(
          child: allEvents.isEmpty
              ? Center(
                  child: Text(
                    showOnlyUnchecked ? '미확인 경보가 없습니다.' : '기록된 이벤트가 없습니다.',
                    style: const TextStyle(color: Color(0xFF4A4E63), fontSize: 12),
                  ),
                )
              : ListView.separated(
                  padding: EdgeInsets.zero,
                  itemBuilder: (context, index) {
                    final event = allEvents[index];
                    return _CombinedAlertTile(event: event, showOnlyUnchecked: showOnlyUnchecked);
                  },
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemCount: allEvents.length,
                ),
        ),
      ],
    );
  }
}

class _CombinedAlertTile extends ConsumerWidget {
  final dynamic event;
  final bool showOnlyUnchecked;

  const _CombinedAlertTile({required this.event, this.showOnlyUnchecked = false});

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
    Color badgeColor;
    String badgeLabel;
    String summaryStr;
    String placeStr;
    bool isChecked = false;

    if (event is Event) {
      isChecked = event.adminChecked == 1 || (event.adminLabel != null && event.adminLabel!.isNotEmpty);
      badgeColor = event.anomalyFlag == 1 ? const Color(0xFFEF4444) : const Color(0xFF38BDF8);
      badgeLabel = event.anomalyFlag == 1 ? 'VISION ALARM' : 'VISION INFO';
      if (isChecked) {
        badgeColor = event.adminLabel == 'normal' ? const Color(0xFF4ADE80) : const Color(0xFF7A7F96);
        badgeLabel = event.adminLabel == 'normal' ? 'NORMAL' : 'CHECKED';
      }
      summaryStr = event.summaryText ?? '비전 이벤트';
      placeStr = event.placeId;
    } else if (event is AudioEvent) {
      isChecked = event.adminChecked == 1 || (event.adminLabel != null && event.adminLabel!.isNotEmpty);
      badgeColor = isChecked ? const Color(0xFF7A7F96) : const Color(0xFFBA68C8);
      badgeLabel = isChecked ? 'CHECKED' : 'AUDIO ALARM';
      summaryStr = event.modelLabel ?? '오디오 이벤트';
      placeStr = '오디오 감시구역';
    } else if (event is YoloEvent) {
      isChecked = event.adminChecked == 1 || (event.adminLabel != null && event.adminLabel!.isNotEmpty);
      badgeColor = isChecked ? const Color(0xFF7A7F96) : const Color(0xFFEAB308);
      badgeLabel = isChecked ? 'CHECKED' : 'YOLO ALARM';
      summaryStr = '${event.eventType ?? "인물 감지"} (${event.personCount}명)';
      placeStr = event.sourceRegionName ?? 'YOLO 감시구역';
    } else if (event is AuthEvent) {
      isChecked = event.adminChecked == 1 || (event.adminLabel != null && event.adminLabel!.isNotEmpty);
      // 2차인증 결과에 따라 배지 색상 구분
      if (isChecked) {
        badgeColor = const Color(0xFF7A7F96);
        badgeLabel = 'CHECKED';
      } else if (event.status == 'fail' || event.status == 'timeout') {
        badgeColor = const Color(0xFFEF4444);
        badgeLabel = 'AUTH ALARM';
      } else {
        badgeColor = const Color(0xFFFACC15);
        badgeLabel = 'AUTH';
      }
      summaryStr = event.resultMessage ?? '2차 인증 이벤트';
      placeStr = event.sourceRegionName ?? '인증 구역';
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
        } else if (event is AuthEvent) {
          showDialog(
            context: context,
            builder: (_) => AuthEventDetailDialog(event: event),
          );
        }
      },
      child: Opacity(
        opacity: isChecked ? 0.6 : 1.0,
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.transparent, // 컨테이너 느낌 제거
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: isChecked
                  ? const Color(0xFF232433)
                  : (showOnlyUnchecked
                      ? const Color(0xFF2D3041)
                      : badgeColor.withOpacity(0.5)),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
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
                  if (event is Event && event.verifiedChangeImageUrl != null && event.verifiedChangeImageUrl!.isNotEmpty) ...[
                    const SizedBox(width: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFACC15).withOpacity(0.14),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: const Text(
                        'HEATMAP',
                        style: TextStyle(color: Color(0xFFFACC15), fontSize: 8, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ],
                  if (isChecked)
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.check_circle, color: Color(0xFF4ADE80), size: 12),
                        const SizedBox(width: 4),
                        Text(
                          (event.adminLabel != null && event.adminLabel!.isNotEmpty) ? event.adminLabel! : '확인됨',
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
              const SizedBox(height: 8),
              Text(
                summaryStr,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: isChecked ? FontWeight.normal : FontWeight.w600,
                  height: 1.3,
                ),
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.access_time, size: 10, color: Color(0xFF757B92)),
                      const SizedBox(width: 4),
                      Text(
                        formatLocalTime(event is Event ? event.capturedAt : event.timestamp),
                        style: const TextStyle(color: Color(0xFF757B92), fontSize: 11),
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
      ),
    );
  }
}
