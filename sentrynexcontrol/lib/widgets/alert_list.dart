import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';
import 'event_detail_dialog.dart';

class AlertList extends ConsumerWidget {
  final bool isCompact;
  final bool showOnlyUnchecked;
  const AlertList({super.key, this.isCompact = false, this.showOnlyUnchecked = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Riverpod 상태에서 이벤트 목록 가져오기
    final allEvents = ref.watch(eventListProvider);
    
    // 조건에 따른 필터링 (showOnlyUnchecked 가 true 일 때만 필터링)
    final alerts = showOnlyUnchecked
        ? allEvents.where((e) {
            final bool isChecked = (e.adminChecked == 1) || (e.adminLabel != null && e.adminLabel!.isNotEmpty);
            return !isChecked;
          }).toList()
        : allEvents;

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: EdgeInsets.all(isCompact ? 12 : 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.warning_amber_outlined,
                    size: isCompact ? 14 : 18,
                    color: const Color(0xFFB5BAD3),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    '실시간 로그',
                    style: TextStyle(
                      color: const Color(0xFFB5BAD3),
                      fontSize: isCompact ? 13 : 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
              // 더미 데이터 테스트용 버튼
              if (!isCompact)
                IconButton(
                  icon: const Icon(Icons.add_circle_outline, color: Color(0xFF26293A), size: 18),
                  onPressed: () {
                    ref.read(eventListProvider.notifier).generateMockEvent();
                  },
                  tooltip: '테스트 알림 추가',
                ),
            ],
          ),
          if (!isCompact) const SizedBox(height: 8),
          if (!isCompact)
            const Text(
              '최근 경보를 확인하려면 더블클릭하세요.',
              style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 12),
            ),
          SizedBox(height: isCompact ? 8 : 16),
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
                      return _AlertTile(event: event, isCompact: isCompact);
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
  final bool isCompact;

  const _AlertTile({required this.event, this.isCompact = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final bool isAnomaly = event.anomalyFlag == 1;
    final bool isChecked = event.adminChecked == 1 || event.adminLabel != null;

    // 배지 상태 정의: 확인된 경우 색상과 라벨 변경 (사용자 요청 반영)
    String badgeLabel = isAnomaly ? 'WARNING' : 'INFO';
    Color badgeColor = isAnomaly ? const Color(0xFFEF4444) : const Color(0xFF38BDF8);

    if (isChecked) {
      if (event.adminLabel == 'normal') {
        badgeLabel = 'NORMAL';
        badgeColor = const Color(0xFF4ADE80); // 초록색 (정상)
      } else {
        badgeLabel = 'CHECKED';
        badgeColor = const Color(0xFF7A7F96); // 회색 (확인 완료된 이상현상)
      }
    }

    return GestureDetector(
      onDoubleTap: () => showEventDetailDialog(context, ref, event),
      child: Opacity(
        opacity: isChecked ? 0.7 : 1.0,
        child: Container(
          padding: EdgeInsets.all(isCompact ? 10 : 12),
          decoration: BoxDecoration(
            color: const Color(0xFF11131C),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: isChecked
                  ? const Color(0xFF2D3041)
                  : (isAnomaly ? const Color(0x33EF4444) : const Color(0xFF2E3244)),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              //---------- 상단 헤더: 상태 배지 및 체크 표시 ----------
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
                        fontSize: isCompact ? 9 : 10,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
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
              //---------- 중간: 이벤트 요약 텍스트 (줄바꿈 원활하도록 Expanded 제거) ----------
              Text(
                event.summaryText ?? '알 수 없는 이벤트',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: isCompact ? 12 : 13,
                  fontWeight: isChecked ? FontWeight.normal : FontWeight.w600,
                  height: 1.3,
                ),
              ),
              const SizedBox(height: 10),
              //---------- 하단: 시간 및 장소 정보 ----------
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.access_time, size: 10, color: Color(0xFF757B92)),
                      const SizedBox(width: 4),
                      Text(
                        formatEventTime(event.capturedAt),
                        style: TextStyle(
                          color: const Color(0xFF757B92),
                          fontSize: isCompact ? 9 : 11,
                        ),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      const Icon(Icons.location_on_outlined, size: 10, color: Color(0xFF757B92)),
                      const SizedBox(width: 4),
                      Text(
                        event.placeId,
                        style: TextStyle(
                          color: const Color(0xFF757B92),
                          fontSize: isCompact ? 9 : 11,
                        ),
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
