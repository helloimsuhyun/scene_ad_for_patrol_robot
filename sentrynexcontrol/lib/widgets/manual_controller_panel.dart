import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ManualControllerPanel extends StatefulWidget {
  const ManualControllerPanel({super.key});

  @override
  State<ManualControllerPanel> createState() => _ManualControllerPanelState();
}

class _ManualControllerPanelState extends State<ManualControllerPanel> {
  bool isManualControlEnabled = false;

  Future<void> _sendCommand(String cmd) async {
    try {
      await http.post(
        Uri.parse('http://127.0.0.1:8000/robot/command'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'command': cmd}),
      );
    } catch (e) {
      debugPrint('Manual Command Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
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
              width: 120, // 십자키 전체 크기
              height: 120,
              child: Stack(
                children: [
                  // 상 (Forward)
                  Align(
                    alignment: Alignment.topCenter,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_up,
                      isEnabled: isManualControlEnabled,
                      onPressed: () => _sendCommand('forward'),
                    ),
                  ),
                  // 하 (Backward)
                  Align(
                    alignment: Alignment.bottomCenter,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_down,
                      isEnabled: isManualControlEnabled,
                      onPressed: () => _sendCommand('backward'),
                    ),
                  ),
                  // 좌 (Left)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_left,
                      isEnabled: isManualControlEnabled,
                      onPressed: () => _sendCommand('turn_left'),
                    ),
                  ),
                  // 우 (Right)
                  Align(
                    alignment: Alignment.centerRight,
                    child: _DirectionButton(
                      icon: Icons.keyboard_arrow_right,
                      isEnabled: isManualControlEnabled,
                      onPressed: () => _sendCommand('turn_right'),
                    ),
                  ),
                  // 중앙 데코 (로봇 아이콘 표시 등)
                  Align(
                    alignment: Alignment.center,
                    child: Container(
                      width: 32,
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
        ],
      ),
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
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: isEnabled ? const Color(0xFF26293A) : const Color(0xFF1C1E2B),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: isEnabled ? const Color(0xFF393C4B) : const Color(0xFF2D3041),
          ),
        ),
        child: Icon(
          icon,
          color: isEnabled ? Colors.white : const Color(0xFF4A4E63),
          size: 20,
        ),
      ),
    );
  }
}
