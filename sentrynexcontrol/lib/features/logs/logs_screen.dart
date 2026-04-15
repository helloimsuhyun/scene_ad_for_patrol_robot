import 'package:flutter/material.dart';
import '../../widgets/alert_list.dart';
import '../../widgets/audio_event_list.dart';
import '../../widgets/yolo_event_list.dart';
import '../../widgets/combined_event_list.dart';

class LogsScreen extends StatelessWidget {
  const LogsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Container(
        color: const Color(0xFF11121A),
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 8),
            Text(
              '데이터베이스에 저장된 비전 / 오디오 / 사람감지 이벤트를 조회합니다 (최신순).',
              style: TextStyle(color: Color(0xFF7A7F96), fontSize: 14),
            ),
            SizedBox(height: 16),
            TabBar(
              indicatorColor: Color(0xFF7F7CFF),
              labelColor: Colors.white,
              unselectedLabelColor: Color(0xFF757B92),
              dividerColor: Colors.transparent,
              tabs: const [
                Tab(text: '전체'),
                Tab(text: '비전 이벤트'),
                Tab(text: '오디오 이벤트'),
                Tab(text: '사람 감지 (YOLO)'),
              ],
            ),
            const SizedBox(height: 24),
            Expanded(
              child: TabBarView(
                children: const [
                  CombinedEventList(showOnlyUnchecked: false),
                  AlertList(showOnlyUnchecked: false),
                  AudioEventList(showOnlyUnchecked: false),
                  YoloEventList(showOnlyUnchecked: false),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
