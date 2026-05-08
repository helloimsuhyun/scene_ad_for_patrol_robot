import 'package:flutter/material.dart';
import 'package:sentrynexcontrol/layout/sidebar.dart';
import 'features/dashboard/dashboard_screen.dart';
import 'features/map/map_screen.dart';
import 'features/logs/logs_screen.dart';
import 'features/settings/settings_screen.dart';
import 'features/control/control_screen.dart';
import 'features/analytics/analytics_screen.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'features/control/calibration_provider.dart';
import 'providers/event_provider.dart';
import 'models/event_model.dart';
import 'widgets/event_detail_dialog.dart';
import 'models/audio_event_model.dart';
import 'models/yolo_event_model.dart';
import 'providers/audio_provider.dart';
import 'providers/yolo_provider.dart';

class MainPage extends ConsumerStatefulWidget {
  const MainPage({super.key});

  @override
  ConsumerState<MainPage> createState() => MainPageState();
}

class MainPageState extends ConsumerState<MainPage> {
  Pages _currentPage = Pages.dashboard;
  String? _lastVisionEventTime;
  String? _lastAudioEventTime;
  String? _lastYoloEventTime;

  final List<Widget> _pages = const [
    DashboardScreen(),
    LogsScreen(),
    AnalyticsScreen(),
    ControlScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    // 글로벌 캘리브레이션 상태 감시
    final calibStatus = ref.watch(calibrationProvider);

    // 글로벌 비정상 이벤트 감지 (스낵바 알림용)
    ref.listen<List<Event>>(eventListProvider, (previous, next) {
      if (next.isNotEmpty) {
        final newEvent = next.first;
        // 시뮬레이터에서 동일 ID를 덮어쓰는 경우 등을 대비해 timestamp(capturedAt)로 비교
        final isNew = _lastVisionEventTime != newEvent.capturedAt;
        
        if (isNew && newEvent.anomalyFlag == 1) {
          _lastVisionEventTime = newEvent.capturedAt;
          _showGlobalAlert(context, ref, 'WARNING', '구역: ${newEvent.placeId}', newEvent);
        }
      }
    });

    ref.listen<List<AudioEvent>>(audioEventListProvider, (previous, next) {
      if (next.isNotEmpty) {
        final newEvent = next.first;
        final isNew = _lastAudioEventTime != newEvent.timestamp;
        
        if (isNew) {
          _lastAudioEventTime = newEvent.timestamp;
          _showGlobalAlert(context, ref, 'AUDIO ALARM', '비정상 음원 감지됨', null, audioEvent: newEvent);
        }
      }
    });

    ref.listen<List<YoloEvent>>(yoloEventsProvider, (previous, next) {
      if (next.isNotEmpty) {
        final newEvent = next.first;
        final isNew = _lastYoloEventTime != newEvent.timestamp;
        
        if (isNew) {
          _lastYoloEventTime = newEvent.timestamp;
          _showGlobalAlert(context, ref, 'YOLO ALARM', '보안 구역 인물 감지', null, yoloEvent: newEvent);
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
              Expanded(
                child: FadeIndexedStack(
                  index: _currentPage.index,
                  children: _pages,
                ),
              ),
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

  void _showGlobalAlert(BuildContext context, WidgetRef ref, String title, String message, Event? visionEvent, {dynamic audioEvent, dynamic yoloEvent}) {
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
                  Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                  Text(message, style: const TextStyle(fontSize: 12, color: Colors.white70)),
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

    if (visionEvent != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        showEventDetailDialog(context, ref, visionEvent);
      });
    } else if (audioEvent != null && audioEvent is AudioEvent) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        showAudioEventDetailDialog(context, ref, audioEvent);
      });
    } else if (yoloEvent != null && yoloEvent is YoloEvent) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        showYoloEventDetailDialog(context, ref, yoloEvent);
      });
    }
  }
}

class FadeIndexedStack extends StatefulWidget {
  final int index;
  final List<Widget> children;
  final Duration duration;

  const FadeIndexedStack({
    super.key,
    required this.index,
    required this.children,
    this.duration = const Duration(milliseconds: 300),
  });

  @override
  State<FadeIndexedStack> createState() => _FadeIndexedStackState();
}

class _FadeIndexedStackState extends State<FadeIndexedStack> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: widget.duration);
    _controller.forward();
  }

  @override
  void didUpdateWidget(FadeIndexedStack oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.index != oldWidget.index) {
      _controller.forward(from: 0.0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _controller,
      child: IndexedStack(
        index: widget.index,
        children: widget.children,
      ),
    );
  }
}
