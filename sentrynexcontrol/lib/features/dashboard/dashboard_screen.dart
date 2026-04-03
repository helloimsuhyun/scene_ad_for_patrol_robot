import 'package:flutter/material.dart';
import 'package:sentrynexcontrol/widgets/dashboard_header.dart';
import 'package:sentrynexcontrol/widgets/data_center_map.dart';
import 'package:sentrynexcontrol/widgets/robot_status_panel.dart';
import 'package:sentrynexcontrol/widgets/camera_stream_widget.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    //---------- 대시보드 전체 화면 배경 설정 ----------
    return Container(
      color: const Color(0xFF11121A),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 대시보드 상단 뼈대: 메인 헤더 ----------
          const DashboardHeader(),
          const SizedBox(height: 16),
          //---------- 대시보드 하단 뼈대: 맵과 사이드 패널 영역 ----------
          Expanded(
            child: Row(
              children: [
                //---------- 하단 좌측 (3/4): 데이터센터 관제 맵 ----------
                Expanded(flex: 3, child: DataCenterMap()),
                const SizedBox(width: 20),
                //---------- 하단 우측 (1/4): 로봇 카메라 및 상태 패널 ----------
                Expanded(
                  flex: 1,
                  child: Column(
                    children: [
                      const CameraStreamWidget(),
                      const SizedBox(height: 20),
                      const Expanded(child: RobotStatusPanel()),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
