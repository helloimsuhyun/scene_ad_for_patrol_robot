import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../providers/event_provider.dart';
import '../../providers/audio_provider.dart';
import '../../providers/yolo_provider.dart';
import '../../providers/server_config_provider.dart';
import '../control/control_provider.dart';
import '../../providers/auth_event_provider.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  String _selectedEventType = 'vision'; // 'vision', 'audio', 'yolo'
  bool _isSimulating = false;
  late TextEditingController _ipController;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController();
    // 초기 로딩 시 현재 IP 설정
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _ipController.text = ref.read(serverConfigProvider).serverIp;
    });
  }

  @override
  void dispose() {
    _ipController.dispose();
    super.dispose();
  }

  void _runSimulation() async {
    setState(() {
      _isSimulating = true;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('시뮬레이션: 3초 후 이벤트가 발생합니다...')),
    );
    
    await Future.delayed(const Duration(seconds: 3));
    if (!mounted) return;

    if (_selectedEventType == 'vision') {
      ref.read(eventListProvider.notifier).generateMockEvent();
    } else if (_selectedEventType == 'audio') {
      ref.read(audioEventListProvider.notifier).injectMockEvent();
    } else if (_selectedEventType == 'yolo') {
      ref.read(yoloEventsProvider.notifier).injectMockEvent();
    }

    setState(() {
      _isSimulating = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF11121A),
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '시스템 설정',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 32),
          
          // 서버 연결 설정 카드
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF181924),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF2D3041)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.settings_remote, color: Color(0xFF4ADE80)),
                    const SizedBox(width: 8),
                    const Text(
                      '서버 연결 설정',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                const Text(
                  '관제 서버(노트북/컴퓨터)의 IP 주소를 입력하세요. (기본: 127.0.0.1)',
                  style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 13),
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _ipController,
                        style: const TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          hintText: '예: 192.168.0.10',
                          hintStyle: const TextStyle(color: Colors.white12),
                          filled: true,
                          fillColor: const Color(0xFF11131C),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: const BorderSide(color: Color(0xFF2D3041)),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: const BorderSide(color: Color(0xFF2D3041)),
                          ),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    ElevatedButton(
                      onPressed: () async {
                        final ip = _ipController.text.trim();
                        if (ip.isNotEmpty) {
                          await ref.read(serverConfigProvider.notifier).setIp(ip);
                          // 모든 실시간 데이터 프로바이더 강제 갱신
                          ref.invalidate(placesProvider);
                          ref.invalidate(eventListProvider);
                          ref.invalidate(audioEventListProvider);
                          ref.invalidate(yoloEventsProvider);
                          ref.invalidate(authEventListProvider);

                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text('서버 IP가 $ip로 업데이트되었습니다. (모든 데이터 갱신됨)')),
                            );
                          }
                        }
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF4ADE80),
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      child: const Text('저장', style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 32),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF181924),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: const Color(0xFF2D3041)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.bug_report_outlined, color: Color(0xFF7F7CFF)),
                    const SizedBox(width: 8),
                    const Text(
                      '이벤트 시뮬레이터 (테스트용)',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                const Text(
                  '선택한 경보 이벤트를 발생시켜 대시보드 상태 변화 및 알림 팝업을 시뮬레이션합니다.',
                  style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 13),
                ),
                const SizedBox(height: 24),
                
                const Text(
                  '경보 종류 선택',
                  style: TextStyle(
                    color: Color(0xFFB5BAD3),
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  decoration: BoxDecoration(
                    color: const Color(0xFF11131C),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFF2D3041)),
                  ),
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      value: _selectedEventType,
                      dropdownColor: const Color(0xFF1C1E2B),
                      isExpanded: true,
                      icon: const Icon(Icons.arrow_drop_down, color: Colors.white54),
                      items: const [
                        DropdownMenuItem(
                          value: 'vision',
                          child: Text('이상 감지 (Vision) - 카메라 강제 진동 감지', style: TextStyle(color: Colors.white)),
                        ),
                        DropdownMenuItem(
                          value: 'audio',
                          child: Text('소리 감지 (Audio) - 유리 깨지는 소리', style: TextStyle(color: Colors.white)),
                        ),
                        DropdownMenuItem(
                          value: 'yolo',
                          child: Text('사람 감지 (YOLO) - 보안 구역 체류 위반', style: TextStyle(color: Colors.white)),
                        ),
                      ],
                      onChanged: _isSimulating
                          ? null
                          : (val) {
                              if (val != null) {
                                setState(() {
                                  _selectedEventType = val;
                                });
                              }
                            },
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _isSimulating ? null : _runSimulation,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1F8CEB),
                      disabledBackgroundColor: const Color(0xFF3D4060),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: _isSimulating
                        ? const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Text(
                            '경보 발생시키기 (3초 후)',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
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
