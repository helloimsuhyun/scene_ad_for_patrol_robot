import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/auth_event_provider.dart';
import '../models/auth_event_model.dart';
import 'auth_event_detail_dialog.dart';

final authEventFilterProvider = StateProvider<String>((ref) => 'all');

class AuthEventList extends ConsumerWidget {
  const AuthEventList({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authEvents = ref.watch(authEventListProvider);

    final currentFilter = ref.watch(authEventFilterProvider);

    // 필터링 로직
    final filteredEvents = authEvents.where((e) {
      if (currentFilter == 'all') {
        // '전체'일 경우: 진행 중(waiting_rfid)이거나 관리자가 확인하지 않은 항목들 우선 표시
        if (e.status == 'waiting_rfid') return true;
        if (e.adminChecked == 0 && (e.adminLabel == null || e.adminLabel!.isEmpty)) return true;
        return false;
      }
      return e.status == currentFilter;
    }).toList();

    final visibleEvents = filteredEvents;

    // 시간 역순 정렬 (최신이 위로)
    visibleEvents.sort((a, b) => b.createdAt.compareTo(a.createdAt));

    if (visibleEvents.isEmpty) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Row(
            children: [
              Icon(Icons.gpp_maybe, color: Color(0xFFFACC15), size: 16),
              SizedBox(width: 6),
              Text(
                '보안 구역 인증 대기',
                style: TextStyle(
                  color: Color(0xFFFACC15),
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(16),
            width: double.infinity,
            decoration: BoxDecoration(
              color: const Color(0xFF181924),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFF2D3041)),
            ),
            alignment: Alignment.center,
            child: const Text(
              '현재 대기 중인 인증 내역이 없습니다.',
              style: TextStyle(color: Color(0xFF4A4E63), fontSize: 12),
            ),
          ),
          const SizedBox(height: 16),
        ],
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Row(
          children: [
            Icon(Icons.gpp_maybe, color: Color(0xFFFACC15), size: 16),
            SizedBox(width: 6),
            Text(
              '보안 구역 인증 대기',
              style: TextStyle(
                color: Color(0xFFFACC15),
                fontSize: 13,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        // ---------- 필터 영역 추가 ----------
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: [
              _buildFilterChip(ref, 'all', '전체'),
              const SizedBox(width: 4),
              _buildFilterChip(ref, 'waiting_rfid', '진행중'),
              const SizedBox(width: 4),
              _buildFilterChip(ref, 'success', '성공'),
              const SizedBox(width: 4),
              _buildFilterChip(ref, 'fail', '실패'),
              const SizedBox(width: 4),
              _buildFilterChip(ref, 'timeout', '미인증'),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Container(
          decoration: BoxDecoration(
            color: const Color(0xFF181924),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: const Color(0xFFFACC15).withOpacity(0.5), width: 1.5),
          ),
          constraints: const BoxConstraints(maxHeight: 200),
          child: ListView.separated(
            padding: const EdgeInsets.all(8),
            shrinkWrap: true,
            itemCount: visibleEvents.length,
            separatorBuilder: (context, index) => const Divider(height: 8, color: Colors.transparent),
            itemBuilder: (context, index) {
              final event = visibleEvents[index];
              return _buildEventItem(context, ref, event);
            },
          ),
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildFilterChip(WidgetRef ref, String status, String label) {
    final currentFilter = ref.watch(authEventFilterProvider);
    final isSelected = currentFilter == status;

    return GestureDetector(
      onTap: () => ref.read(authEventFilterProvider.notifier).state = status,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFFFACC15).withOpacity(0.2) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? const Color(0xFFFACC15) : const Color(0xFF2D3041),
            width: 1,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? const Color(0xFFFACC15) : const Color(0xFF9FA4B9),
            fontSize: 10,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }

  Widget _buildEventItem(BuildContext context, WidgetRef ref, AuthEvent event) {
    Color statusColor;
    IconData statusIcon;
    String statusText;

    switch (event.status) {
      case 'waiting_rfid':
        statusColor = const Color(0xFFFACC15);
        statusIcon = Icons.hourglass_top;
        statusText = '진행중';
        break;
      case 'success':
        statusColor = const Color(0xFF4ADE80);
        statusIcon = Icons.check_circle_outline;
        statusText = '성공';
        break;
      case 'fail':
        statusColor = const Color(0xFFEF4444);
        statusIcon = Icons.error_outline;
        statusText = '실패';
        break;
      case 'timeout':
      default:
        statusColor = const Color(0xFF9FA4B9);
        statusIcon = Icons.timer_off_outlined;
        statusText = '미인증';
        break;
    }

    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () {
          showDialog(
            context: context,
            builder: (ctx) => AuthEventDetailDialog(event: event),
          );
        },
        child: Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: const Color(0xFF232433),
            borderRadius: BorderRadius.circular(8),
            border: event.status == 'waiting_rfid' 
                ? Border.all(color: statusColor.withOpacity(0.5))
                : null,
          ),
          child: Row(
            children: [
              Icon(statusIcon, color: statusColor, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          statusText,
                          style: TextStyle(
                            color: statusColor,
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          event.sourceRegionName ?? '알 수 없는 구역',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      event.timestamp.split('T').last.split('.').first,
                      style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: Color(0xFF4A4E63), size: 16),
            ],
          ),
        ),
      ),
    );
  }
}
