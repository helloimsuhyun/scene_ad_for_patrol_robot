import 'package:flutter/material.dart';

class RobotStatusPanel extends StatefulWidget {
  const RobotStatusPanel({super.key});

  @override
  State<RobotStatusPanel> createState() => _RobotStatusPanelState();
}

class _RobotStatusPanelState extends State<RobotStatusPanel> {
  bool isManualControlEnabled = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12), // 패널 안쪽 여백 줄임 (로그창 공간 확보)
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Icon(
                Icons.smart_toy_outlined,
                size: 18,
                color: Color(0xFFB5BAD3),
              ),
              const SizedBox(width: 6),
              const Text(
                'Robot Status',
                style: TextStyle(
                  color: Color(0xFFB5BAD3),
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              //---------- 우상단 소형 배터리 인디케이터 ----------
              Row(
                children: [
                  Container(
                    width: 60, // 배터리 막대 가로 늘림
                    height: 14, // 세로 늘림
                    decoration: BoxDecoration(
                      color: const Color(0xFF26293A),
                      borderRadius: BorderRadius.circular(5),
                    ),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Container(
                        width: 60 * 0.72,
                        decoration: BoxDecoration(
                          color: const Color(0xFF7F7CFF), // 포인트 컬러 연보라색
                          borderRadius: BorderRadius.circular(5),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    '72%',
                    style: TextStyle(
                      color: Color(0xFF7F7CFF), // 연보라색 텍스트
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              _RobotMetric(
                label: '연결 상태',
                value: '온라인',
                color: Color(0xFF4ADE80),
              ),
              _RobotMetric(
                label: '현재 속도',
                value: '1.2 m/s',
                color: Color(0xFFB5BAD3),
              ),
              _RobotMetric(
                label: '작업 모드',
                value: '순찰 중',
                color: Color(0xFFB5BAD3),
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Divider(height: 1, color: Color(0xFF2D3041)),
          const SizedBox(height: 10),
          //---------- 수동 조작 컨트롤러 헤더 및 스위치 ----------
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '수동 조작 컨트롤러',
                style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
              ),
              Switch(
                value: isManualControlEnabled,

                inactiveThumbColor: const Color(0xFF7F7CFF),
                activeColor: const Color(0xFF7F7CFF),
                onChanged: (value) {
                  setState(() {
                    isManualControlEnabled = value;
                  });
                },
              ),
            ],
          ),
          const SizedBox(height: 6),
          //---------- 십자키 조이스틱 ----------
          Center(
            child: SizedBox(
              width: 120, // 십자키 전체 크기 지정 (축소)
              height: 120,
              child: Stack(
                children: [
                  // 상 (Forward)
                  Align(
                    alignment: Alignment.topCenter,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_up,
                      isEnabled: isManualControlEnabled,
                      onPressed: () {},
                    ),
                  ),
                  // 하 (Backward)
                  Align(
                    alignment: Alignment.bottomCenter,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_down,
                      isEnabled: isManualControlEnabled,
                      onPressed: () {},
                    ),
                  ),
                  // 좌 (Left)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_left,
                      isEnabled: isManualControlEnabled,
                      onPressed: () {},
                    ),
                  ),
                  // 우 (Right)
                  Align(
                    alignment: Alignment.centerRight,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_right,
                      isEnabled: isManualControlEnabled,
                      onPressed: () {},
                    ),
                  ),
                  // 중앙 데코 (로봇 아이콘 표시 등)
                  Align(
                    alignment: Alignment.center,
                    child: Container(
                      width: 32, // 중앙 데코 크기 축소
                      height: 32,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isManualControlEnabled
                            ? const Color(0xFF7F7CFF)
                            : const Color(0xFF2D3041),
                        boxShadow: isManualControlEnabled
                            ? const [
                                BoxShadow(
                                  color: Color(0x667F7CFF),
                                  blurRadius: 8,
                                  spreadRadius: 2,
                                ),
                              ]
                            : null,
                      ),
                      child: Icon(
                        Icons.smart_toy,
                        color: isManualControlEnabled
                            ? Colors.white
                            : const Color(0xFF6B7280),
                        size: 16,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}

class _RobotMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;

  const _RobotMetric({required this.label, required this.value, this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color ?? Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

//---------- 개별 방향 전환용 버튼 위젯 ----------
class _DirectionButton extends StatelessWidget {
  final IconData icon;
  final bool isEnabled;
  final VoidCallback onPressed;

  const _DirectionButton({
    required this.icon,
    required this.isEnabled,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isEnabled ? onPressed : null,
      child: Container(
        width: 38, // 십자키 개별 버튼 크기 축소 44 -> 38
        height: 38,
        decoration: BoxDecoration(
          color: isEnabled ? const Color(0xFF26293A) : const Color(0xFF1C1E2B),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: isEnabled
                ? const Color(0xFF393C4B)
                : const Color(0xFF2D3041),
          ),
        ),
        child: Icon(
          icon,
          color: isEnabled ? Colors.white : const Color(0xFF4A4E63),
          size: 20, // 아이콘 크기 축소 24 -> 20
        ),
      ),
    );
  }
}
