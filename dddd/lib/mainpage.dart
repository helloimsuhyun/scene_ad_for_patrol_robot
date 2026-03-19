import 'package:flutter/material.dart';
import 'package:sentrynexcontrol/layout/sidebar.dart';
import 'features/dashboard/dashboard_screen.dart';
import 'features/map/map_screen.dart';
import 'features/logs/logs_screen.dart';
import 'features/settings/settings_screen.dart';

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => MainPageState();
}

class MainPageState extends State<MainPage> {
  Pages currentPage = Pages.dashboard;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          //---------- 좌측 사이드바 고정 너비 영역 ----------
          SizedBox(
            width: 232, // 고정 치수를 주어 텍스트 오버플로우 방지
            child: Sidebar(
              currentPage: currentPage,
              onPageSelected: (page) {
                setState(() {
                  currentPage = page;
                });
              },
            ),
          ),
          //---------- 우측 메인 콘텐츠 영역 ----------
          Expanded(child: buildPage()),
        ],
      ),
    );
  }

  Widget buildPage() {
    switch (currentPage) {
      case Pages.dashboard:
        return const DashboardScreen();
      case Pages.map:
        return const MapScreen();
      case Pages.logs:
        return const LogsScreen();
      case Pages.settings:
        return const SettingsScreen();
    }
  }
}
