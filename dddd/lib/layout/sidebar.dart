import 'package:flutter/material.dart';
import 'sidebar_menu.dart';
import '../widgets/camera_stream_widget.dart';

enum Pages { dashboard, map, logs, settings }

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
            isSelected: currentPage == Pages.map,
            onTap: () => onPageSelected(Pages.map),
          ),
          const SizedBox(height: 6),
          SidebarMenu(
            title: '설정',
            icon: Icons.settings_outlined,
            isSelected: currentPage == Pages.settings,
            onTap: () => onPageSelected(Pages.settings),
          ),
          //---------- 메뉴 아래: 로봇 카메라 실시간 스트림 ----------
          const SizedBox(height: 20),
          const CameraStreamWidget(),
          const Spacer(),
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
