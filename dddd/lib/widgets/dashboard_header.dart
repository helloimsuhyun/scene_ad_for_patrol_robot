import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:sentrynexcontrol/features/dashboard/dashboard_provider.dart';

class DashboardHeader extends ConsumerWidget {
  const DashboardHeader({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final locations = const ['데이터센터 3층 전산실', '데이터센터 2층 서버룸', '제1 센터 외부 구역'];
    final selectedLocation = ref.watch(dashboardMapLocationProvider);

    //---------- 대시보드 상단 해더 레이아웃 ----------
    return Row(
      children: [
        //---------- 해더 좌측: 데이터센터 위치 선택 드롭다운 ----------
        _MapLocationMenu(
          value: selectedLocation,
          items: locations,
          onSelected: (value) =>
              ref.read(dashboardMapLocationProvider.notifier).state = value,
        ),
        const Spacer(),
        //---------- 해더 우측: 서버 상태 표시 ----------
        Row(
          children: [
            //---------- 상태 표시 아이콘 ----------
            Icon(Icons.cloud_done, size: 16, color: Color(0xFF7F7CFF)),
            const SizedBox(width: 8),
            //---------- 서버 상태 텍스트 ----------
            const Text(
              '서버 연결됨',
              style: TextStyle(
                color: Color(0xFF9FA4B9),
                fontSize: 14, // 크기 살짝 키움
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(width: 12),
            IconButton(
              onPressed: () {},
              icon: Icon(Icons.notifications, size: 20, color: Colors.grey),
            ),
          ],
        ),
      ],
    );
  }
}

class _MapLocationMenu extends StatelessWidget {
  final String value;
  final List<String> items;
  final ValueChanged<String> onSelected;

  const _MapLocationMenu({
    required this.value,
    required this.items,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    //---------- 위치 선택 컴포넌트: 전체 감싸는 MenuAnchor ----------
    return MenuAnchor(
      style: MenuStyle(
        backgroundColor: const WidgetStatePropertyAll<Color>(Color(0xFF11131C)),
        elevation: const WidgetStatePropertyAll<double>(12),
        shape: WidgetStatePropertyAll<RoundedRectangleBorder>(
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        side: const WidgetStatePropertyAll<BorderSide>(
          BorderSide(color: Color(0xFF3C3F52)),
        ),
        padding: const WidgetStatePropertyAll<EdgeInsets>(
          EdgeInsets.symmetric(vertical: 6),
        ),
      ),
      builder: (context, controller, _) {
        //---------- 드롭다운 트리거 버튼 부분 (선택된 구역 텍스트 + 화살표) ----------
        return InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: () {
            if (controller.isOpen) {
              controller.close();
            } else {
              controller.open();
            }
          },
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: const Color(0xFF11131C),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFF3C3F52)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  value,
                  style: const TextStyle(
                    color: Color(0xFFB5BAD3),
                    fontSize: 12,
                  ),
                ),
                const SizedBox(width: 6),
                const Icon(
                  Icons.expand_more,
                  color: Color(0xFFB5BAD3),
                  size: 18,
                ),
              ],
            ),
          ),
        );
      },
      //---------- 드롭다운 열렸을 때 나오는 메뉴 아이템들 ----------
      menuChildren: [
        for (final item in items)
          MenuItemButton(
            style: const ButtonStyle(
              padding: WidgetStatePropertyAll(
                EdgeInsets.symmetric(horizontal: 14),
              ),
            ),
            onPressed: () => onSelected(item),
            child: Text(
              item,
              style: TextStyle(
                color: item == value ? Colors.white : const Color(0xFFB5BAD3),
                fontSize: 13,
                fontWeight: item == value ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ),
      ],
    );
  }
}
