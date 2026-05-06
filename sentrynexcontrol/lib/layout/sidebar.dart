import 'package:flutter/material.dart';
import 'sidebar_menu.dart';
import '../widgets/combined_event_list.dart';
import '../widgets/auth_event_list.dart';

enum Pages { dashboard, logs, analytics, control, settings }

class Sidebar extends StatelessWidget {
  final Pages currentPage;
  final ValueChanged<Pages> onPageSelected;

  const Sidebar({
    super.key,
    required this.currentPage,
    required this.onPageSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFF11121A),
        border: Border(right: BorderSide(color: Color(0xFF232433), width: 1.0)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.shield, size: 18, color: Colors.white),
              const SizedBox(width: 6),
              Text(
                'SENTRYNEX Control.',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w400,
                ),
              ),
            ],
          ),
          const SizedBox(height: 32),
          Transform.translate(
            offset: const Offset(4, 0),
            child: Text(
              '메뉴',
              style: TextStyle(
                color: const Color(0xFF7A7F96),
                fontSize: 12,
                fontWeight: FontWeight.w400,
              ),
            ),
          ),

          const SizedBox(height: 14),
          SidebarMenu(
            title: '대시보드',
            icon: Icons.grid_view_outlined,
            isSelected: currentPage == Pages.dashboard,
            onTap: () => onPageSelected(Pages.dashboard),
          ),
          const SizedBox(height: 6),
          SidebarMenu(
            title: '경보 기록',
            icon: Icons.format_list_bulleted_sharp,
            isSelected: currentPage == Pages.logs,
            onTap: () => onPageSelected(Pages.logs),
          ),
          const SizedBox(height: 6),
          SidebarMenu(
            title: '분석',
            icon: Icons.bar_chart,
            isSelected: currentPage == Pages.analytics,
            onTap: () => onPageSelected(Pages.analytics),
          ),
          const SizedBox(height: 6),
          SidebarMenu(
            title: '제어',
            icon: Icons.compare_arrows,
            isSelected: currentPage == Pages.control,
            onTap: () => onPageSelected(Pages.control),
          ),
          const SizedBox(height: 6),
          SidebarMenu(
            title: '설정',
            icon: Icons.settings_outlined,
            isSelected: currentPage == Pages.settings,
            onTap: () => onPageSelected(Pages.settings),
          ),
          //---------- 메뉴 아래: 2차 인증 및 실시간 경보 로그 ----------
          const SizedBox(height: 36),
          const Divider(height: 1, color: Color(0xFF2D3041)),
          const SizedBox(height: 20),
          const AuthEventList(), // 2차 인증 이벤트 리스트
          const Expanded(
            child: CombinedEventList(showOnlyUnchecked: true, title: '실시간 로그'),
          ),
          const SizedBox(height: 36),
          //---------- 카메라 아래: 수동 조작 컨트롤러 (UI 임시 제거) ----------
          // const ManualControllerPanel(),
          // const SizedBox(height: 16),
          //---------- 사이드바 하단 접기/열기 버튼 ----------
          Align(
            alignment: Alignment.centerRight,
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(8),
                onTap: () {
                  // TODO: 사이드바 토글 기능 추가
                },
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: const [
                      Text(
                        '접기',
                        style: TextStyle(
                          color: Color(0xFF7A7F96),
                          fontSize: 12,
                        ),
                      ),
                      SizedBox(width: 4),
                      Icon(
                        Icons.keyboard_double_arrow_left,
                        color: Color(0xFF7A7F96),
                        size: 20,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
