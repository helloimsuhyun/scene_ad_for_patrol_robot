import 'package:flutter/material.dart';
import 'package:sentrynexcontrol/layout/sidebar.dart';
import 'features/dashboard/dashboard_screen.dart';
import 'features/map/map_screen.dart';
import 'features/logs/logs_screen.dart';
import 'features/settings/settings_screen.dart';
import 'features/control/control_screen.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'features/control/calibration_provider.dart';
import 'providers/event_provider.dart';
import 'models/event_model.dart';

class MainPage extends ConsumerStatefulWidget {
  const MainPage({super.key});

  @override
  ConsumerState<MainPage> createState() => MainPageState();
}

class MainPageState extends ConsumerState<MainPage> {
  Pages _currentPage = Pages.dashboard;

  @override
  Widget build(BuildContext context) {
    // 글로벌 캘리브레이션 상태 감시
    final calibStatus = ref.watch(calibrationProvider);

    // 글로벌 비정상 이벤트 감지 (스낵바 알림용)
    ref.listen<List<Event>>(eventListProvider, (previous, next) {
      if (previous != null && next.isNotEmpty && next.length > previous.length) {
        // Assuming the new event is added to the beginning of the list
        final newEvent = next.first;
        if (newEvent.anomalyFlag == 1) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 28),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text('위험 감지! (구역: ${newEvent.placeId})', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                        Text('시스템이 새로운 비정상 이벤트를 감지했습니다.', style: const TextStyle(fontSize: 12, color: Colors.white70)),
                      ],
                    ),
                  ),
                ],
              ),
              backgroundColor: Colors.redAccent.shade700,
              behavior: SnackBarBehavior.floating,
              margin: const EdgeInsets.only(bottom: 20, left: 20, right: 20),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              duration: const Duration(seconds: 4),
            ),
          );
        }
      }
    });



    return Scaffold(
      body: Stack(
        children: [
          Row(
            children: [
              //---------- 좌측 사이드바 고정 너비 영역 ----------
              SizedBox(
                width: 232, // 고정 치수를 주어 텍스트 오버플로우 방지
                child: Sidebar(
                  currentPage: _currentPage,
                  onPageSelected: (page) {
                    setState(() {
                      _currentPage = page;
                    });
                  },
                ),
              ),
              //---------- 우측 메인 콘텐츠 영역 ----------
              Expanded(child: buildPage()),
            ],
          ),
          
          //---------- 딥러닝 연산 대기용 전체화면 오버레이 ----------
          if (calibStatus.globalCalibrating)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(0.8), // 주변 블락
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const CircularProgressIndicator(color: Color(0xFF1F8CEB)),
                      const SizedBox(height: 24),
                      const Text(
                        '딥러닝 시스템 재학습 중입니다...',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          decoration: TextDecoration.none,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '잠시만 기다려주세요 (${calibStatus.done}/${calibStatus.total})',
                        style: const TextStyle(
                          color: Color(0xFF9FA4B9),
                          fontSize: 14,
                          decoration: TextDecoration.none,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget buildPage() {
    switch (_currentPage) {
      case Pages.dashboard:
        return const DashboardScreen();
      case Pages.map:
        return const MapScreen();
      case Pages.logs:
        return const LogsScreen();
      case Pages.control:
        return const ControlScreen();
      case Pages.settings:
        return const SettingsScreen();
      default:
        return const SizedBox.shrink();
    }
  }
}
