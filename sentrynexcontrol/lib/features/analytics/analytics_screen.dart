import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../providers/event_provider.dart';
import '../../providers/audio_provider.dart';
import '../../providers/yolo_provider.dart';
import '../../models/event_model.dart';
import '../../models/audio_event_model.dart';
import '../../models/yolo_event_model.dart';

class AnalyticsScreen extends ConsumerWidget {
  const AnalyticsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 1. 데이터 가져오기
    final visionEvents = ref.watch(eventListProvider);
    final audioEvents = ref.watch(audioEventListProvider);
    final yoloEvents = ref.watch(yoloEventsProvider);

    // 2. 당일 발생 데이터 필터링 로직 (여기서는 데모를 위해 전체를 당일이라 가정)
    // 실제로는 timestamp를 파싱해서 DateTime.now()와 오늘 날짜인지 비교해야 함.

    final int visionCount = visionEvents.length;
    final int audioCount = audioEvents.length;
    final int yoloCount = yoloEvents.length;
    final int totalAlerts = visionCount + audioCount + yoloCount;

    // 미확인 경보 (admin_checked == 0) 필터링
    final int uncheckedVision = visionEvents.where((e) => e.adminChecked == 0).length;
    final int uncheckedAudio = audioEvents.where((e) => e.adminChecked == 0).length;
    final int uncheckedYolo = yoloEvents.where((e) => e.adminChecked == 0).length;
    final int totalUnchecked = uncheckedVision + uncheckedAudio + uncheckedYolo;

    //---------- 전체 화면 구성 ----------
    return Container(
      color: const Color(0xFF0F1015), // 배경색
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 헤더 텍스트 ----------
          const Text(
            'Analytics Dashboard',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            '당일 발생한 이벤트 통계를 한눈에 확인하세요.',
            style: TextStyle(
              color: Color(0xFF7A7F96),
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 24),

          //---------- 상단 KPI 요약 카드 ----------
          SizedBox(
            width: 300,
            child: _KpiCard(
              title: '오늘 발생 알림',
              value: '$totalAlerts 건',
              valueColor: Colors.white,
              icon: Icons.notifications_active_outlined,
              iconColor: const Color(0xFF38BDF8),
            ),
          ),
          const SizedBox(height: 48),

          //---------- 하단 차트 영역 ----------
          SizedBox(
            height: 360,
            child: Row(
              children: [
                // 좌측: 카테고리별 도넛 차트
                Expanded(
                  flex: 1,
                  child: Container(
                    decoration: BoxDecoration(
                      color: const Color(0xFF181924),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF2D3041)),
                    ),
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          '알림 카테고리 비율',
                          style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 16),
                        Expanded(
                          child: PieChart(
                            PieChartData(
                              sectionsSpace: 4,
                              centerSpaceRadius: 60,
                              sections: _buildPieSections(visionCount, audioCount, yoloCount),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        // 범례
                        _buildLegendList(),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                // 우측: 시간대별 알림 발생 (막대그래프)
                Expanded(
                  flex: 2,
                  child: Container(
                    decoration: BoxDecoration(
                      color: const Color(0xFF181924),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF2D3041)),
                    ),
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          '시간대별 경보 발생 추이',
                          style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 24),
                        Expanded(
                          child: BarChart(
                            BarChartData(
                              alignment: BarChartAlignment.spaceAround,
                              maxY: 10, // 임시 최대값
                              barTouchData: BarTouchData(enabled: false),
                              titlesData: FlTitlesData(
                                show: true,
                                bottomTitles: AxisTitles(
                                  sideTitles: SideTitles(
                                    showTitles: true,
                                    getTitlesWidget: (value, meta) {
                                      return Padding(
                                        padding: const EdgeInsets.only(top: 8.0),
                                        child: Text(
                                          '${value.toInt()}h',
                                          style: const TextStyle(color: Color(0xFF7A7F96), fontSize: 12),
                                        ),
                                      );
                                    },
                                  ),
                                ),
                                leftTitles: AxisTitles(
                                  sideTitles: SideTitles(
                                    showTitles: true,
                                    reservedSize: 28,
                                    interval: 1, // 정수 간격으로 표시
                                    getTitlesWidget: (value, meta) {
                                      // 소수점인 경우 빈 텍스트 반환
                                      if (value != value.toInt().toDouble()) return const SizedBox.shrink();
                                      return Text(
                                        value.toInt().toString(),
                                        style: const TextStyle(color: Color(0xFF7A7F96), fontSize: 12),
                                      );
                                    },
                                  ),
                                ),
                                topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                              ),
                              gridData: FlGridData(
                                show: true,
                                drawVerticalLine: false,
                                getDrawingHorizontalLine: (value) => FlLine(
                                  color: const Color(0xFF2D3041),
                                  strokeWidth: 1,
                                ),
                              ),
                              borderData: FlBorderData(show: false),
                              barGroups: _buildMockBarGroups(),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          )
        ],
      ),
    );
  }

  //---------- 통계 차트를 위한 데이터 모델 및 UI 헬퍼 ----------

  // 파이 차트 섹션 생성 (시각화를 위해 각 알림 모델 종류별로 도넛 비율 생성)
  List<PieChartSectionData> _buildPieSections(int vision, int audio, int yolo) {
    // 0일 경우 기본값을 보여주기 위함 (시각적인 예시)
    if (vision == 0 && audio == 0 && yolo == 0) {
      return [
        PieChartSectionData(
          color: const Color(0xFF3D4060),
          value: 1,
          title: '',
          radius: 20,
        )
      ];
    }

    return [
      if (vision > 0)
        PieChartSectionData(
          color: const Color(0xFFF43F5E), // Vision Red
          value: vision.toDouble(),
          title: '$vision',
          radius: 25,
          titleStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white),
        ),
      if (audio > 0)
        PieChartSectionData(
          color: const Color(0xFF7F7CFF), // Audio Purple
          value: audio.toDouble(),
          title: '$audio',
          radius: 25,
          titleStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white),
        ),
      if (yolo > 0)
        PieChartSectionData(
          color: const Color(0xFFEAB308), // Yolo Yellow
          value: yolo.toDouble(),
          title: '$yolo',
          radius: 25,
          titleStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white),
        ),
    ];
  }

  // 차트 범례 UI
  Widget _buildLegendList() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _Indicator(color: const Color(0xFFF43F5E), text: 'Vision'),
        const SizedBox(width: 12),
        _Indicator(color: const Color(0xFF7F7CFF), text: 'Audio'),
        const SizedBox(width: 12),
        _Indicator(color: const Color(0xFFEAB308), text: 'YOLO'),
      ],
    );
  }

  // 임시 막대차트 데이터 (이후 실제 로직에서 Provider의 timestamp 기준 병합 처리 요망)
  List<BarChartGroupData> _buildMockBarGroups() {
    double barWidth = 14;
    return [
      BarChartGroupData(x: 9, barRods: [BarChartRodData(toY: 1, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
      BarChartGroupData(x: 10, barRods: [BarChartRodData(toY: 3, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
      BarChartGroupData(x: 11, barRods: [BarChartRodData(toY: 5, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
      BarChartGroupData(x: 12, barRods: [BarChartRodData(toY: 2, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
      BarChartGroupData(x: 13, barRods: [BarChartRodData(toY: 8, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
      BarChartGroupData(x: 14, barRods: [BarChartRodData(toY: 4, width: barWidth, color: const Color(0xFF38BDF8), borderRadius: BorderRadius.circular(4))]),
    ];
  }
}

// KPI 카드 공통 UI
class _KpiCard extends StatelessWidget {
  final String title;
  final String value;
  final Color valueColor;
  final IconData icon;
  final Color iconColor;

  const _KpiCard({required this.title, required this.value, required this.valueColor, required this.icon, required this.iconColor});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: iconColor, size: 28),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(color: Color(0xFF7A7F96), fontSize: 13)),
              const SizedBox(height: 6),
              Text(value, style: TextStyle(color: valueColor, fontSize: 22, fontWeight: FontWeight.bold)),
            ],
          )
        ],
      ),
    );
  }
}

// 범례 항목 UI 위젯
class _Indicator extends StatelessWidget {
  final Color color;
  final String text;

  const _Indicator({required this.color, required this.text});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(width: 12, height: 12, decoration: BoxDecoration(shape: BoxShape.circle, color: color)),
        const SizedBox(width: 6),
        Text(text, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }
}
