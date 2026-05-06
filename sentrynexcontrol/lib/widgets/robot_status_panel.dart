import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../providers/robot_provider.dart';
import '../providers/map_provider.dart';
import '../features/control/control_provider.dart'; // placesProvider 참조
import 'patrol_route_dialog.dart';
import 'timeline_dialog.dart';

// 로봇 현재 목표 지점 폴링은 실제 연동 시 활성화 예정
// final _robotGoalProvider = StreamProvider...

// 수동 캡처 대기를 위한 로딩 상태
final _isCapturingProvider = StateProvider<bool>((ref) => false);

// 수동 캡처 테스트를 위한 현재 라벨 상태 (z 명령어 토글용)
final _queryLabelProvider = StateProvider<String>((ref) => 'normal');

// 수동 캡처(이동+캡처)를 위한 테스트용 타겟 구역
final _testTargetPlaceProvider = StateProvider<String?>((ref) => null);

class RobotStatusPanel extends ConsumerStatefulWidget {
  const RobotStatusPanel({super.key});

  @override
  ConsumerState<RobotStatusPanel> createState() => _RobotStatusPanelState();
}

class _RobotStatusPanelState extends ConsumerState<RobotStatusPanel> {
  bool isCliButtonsEnabled = false;
  // 로컬 _isPatrolling 제거 (전역 patrolStatusProvider 사용)

  Future<void> _triggerCapture(
    BuildContext context,
    WidgetRef ref,
    String endpoint,
  ) async {
    final loadingNotifier = ref.read(_isCapturingProvider.notifier);
    loadingNotifier.state = true;
    try {
      final targetPlace = ref.read(_testTargetPlaceProvider);
      final body = endpoint == 'place_and_capture' && targetPlace != null
          ? jsonEncode({'place_id': targetPlace})
          : null;

      final response = await http.post(
        Uri.parse('http://192.168.0.24:8090/patrol/$endpoint'),
        headers: body != null ? {'Content-Type': 'application/json'} : null,
        body: body,
      );
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['image_b64'] != null && context.mounted) {
          final b64String = data['image_b64'].toString().replaceFirst(
            RegExp(r'data:image/[^;]+;base64,'),
            '',
          );
          final bytes = base64Decode(b64String);
          showDialog(
            context: context,
            builder: (ctx) => Dialog(
              backgroundColor: const Color(0xFF1C1E2B),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      '📸 캡처 완료!',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.memory(
                        bytes,
                        height: 300,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 16),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF1F8CEB),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 12),
                        ),
                        onPressed: () => Navigator.of(ctx).pop(),
                        child: const Text(
                          '확인',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }
      } else {
        if (context.mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(const SnackBar(content: Text('캡처 명령 실패')));
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('에러 발생: $e')));
      }
    } finally {
      loadingNotifier.state = false;
    }
  }

  Future<void> _sendRobotCommand(String cmd) async {
    debugPrint('Robot Command Received: $cmd');
    // 순찰 시작/정지 버튼에 따라 전역 순찰 상태 즉시 반영
    if (cmd == 'start_patrol') {
      debugPrint('Setting patrolStatusProvider to TRUE');
      ref.read(patrolStatusProvider.notifier).state = true;
    } else if (cmd == 'pause_patrol' || cmd == 'return_to_charge') {
      debugPrint('Setting patrolStatusProvider to FALSE');
      ref.read(patrolStatusProvider.notifier).state = false;
    }
    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8000/robot/command'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'command': cmd}),
      );
      if (res.statusCode == 200 && mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('명령 전송: $cmd')));
      }
    } catch (e) {
      debugPrint('Command err: $e');
    }
  }

  Future<void> _toggleQueryLabel(WidgetRef ref) async {
    final current = ref.read(_queryLabelProvider);
    final next = current == 'normal' ? 'abnormal' : 'normal';

    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8000/query_capture_label'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'label': next}),
      );

      if (res.statusCode != 200) {
        throw Exception('failed: ${res.statusCode} ${res.body}');
      }

      final data = jsonDecode(res.body) as Map<String, dynamic>;
      final serverLabel = (data['query_capture_label'] ?? next).toString();

      ref.read(_queryLabelProvider.notifier).state = serverLabel;
      debugPrint('[LABEL] updated -> $serverLabel');
    } catch (e) {
      debugPrint('Error toggling label: $e');
    }
  }

  void _toggleWaypointPickingMode() {
    final current = ref.read(waypointPickingModeProvider);
    ref.read(waypointPickingModeProvider.notifier).state = !current;
  }

  @override
  Widget build(BuildContext context) {
    final robotPose = ref.watch(robotPoseProvider);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12), // 패널 안쪽 여백 줄임 (로그창 공간 확보)
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      child: SingleChildScrollView(
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
              children: [
                const _RobotMetric(
                  label: '연결 상태',
                  value: '온라인',
                  color: Color(0xFF4ADE80),
                ),
                const _RobotMetric(
                  label: '현재 속도',
                  value: '1.2 m/s',
                  color: Color(0xFFB5BAD3),
                ),
                _RobotMetric(
                  label: '작업 모드',
                  value: robotPose?.status ?? '알 수 없음',
                  color: const Color(0xFFB5BAD3),
                ),
              ],
            ),
            const SizedBox(height: 14),
            const Divider(height: 1, color: Color(0xFF2D3041)),
            const SizedBox(height: 10),
            //---------- 명령어 컨트롤러 헤더 및 스위치 (위치 이동됨) ----------
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '명령어 컨트롤러',
                  style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
                ),
                Switch(
                  value: isCliButtonsEnabled,
                  inactiveThumbColor: const Color(0xFF7F7CFF),
                  activeColor: const Color(0xFF7F7CFF),
                  onChanged: (value) {
                    setState(() {
                      isCliButtonsEnabled = value;
                    });
                  },
                ),
              ],
            ),

            if (isCliButtonsEnabled) ...[
              const SizedBox(height: 8),
              Builder(
                builder: (context) {
                  final queryLabel = ref.watch(_queryLabelProvider);
                  final isCapturing = ref.watch(_isCapturingProvider);
                  final testPlace = ref.watch(_testTargetPlaceProvider);
                  final placesAsync = ref.watch(placesProvider);

                  return Column(
                    children: [
                      // ---------- 1열: 라벨 및 대상 구역 선택 ----------
                      Row(
                        children: [
                          Expanded(
                            child: TextButton.icon(
                              onPressed: () => _toggleQueryLabel(ref),
                              style: TextButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 8,
                                ),
                                minimumSize: Size.zero,
                                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                              ),
                              icon: Icon(
                                queryLabel == 'normal'
                                    ? Icons.shield_outlined
                                    : Icons.warning_amber_rounded,
                                color: queryLabel == 'normal'
                                    ? Colors.greenAccent
                                    : Colors.redAccent,
                                size: 13,
                              ),
                              label: Text(
                                '라벨: $queryLabel',
                                style: TextStyle(
                                  color: queryLabel == 'normal'
                                      ? Colors.greenAccent
                                      : Colors.redAccent,
                                  fontSize: 10,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            flex: 2,
                            child: Container(
                              height: 32,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                              ),
                              decoration: BoxDecoration(
                                color: const Color(0xFF26293A),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: DropdownButtonHideUnderline(
                                child: placesAsync.when(
                                  data: (data) {
                                    final list =
                                        data['places'] as List<dynamic>? ?? [];
                                    // 현재 선택된 값이 list에 없으면 null
                                    final currentVal =
                                        list.any(
                                          (p) =>
                                              p['place_id'].toString() ==
                                              testPlace,
                                        )
                                        ? testPlace
                                        : null;
                                    return DropdownButton<String>(
                                      value: currentVal,
                                      hint: const Text(
                                        '테스트 이동 구역 선택',
                                        style: TextStyle(
                                          color: Colors.white54,
                                          fontSize: 10,
                                        ),
                                      ),
                                      dropdownColor: const Color(0xFF1C1E2B),
                                      isExpanded: true,
                                      icon: const Icon(
                                        Icons.arrow_drop_down,
                                        color: Colors.white54,
                                        size: 16,
                                      ),
                                      items: list.map((p) {
                                        final pid = p['place_id'].toString();
                                        final name =
                                            p['display_name']?.toString() ??
                                            pid;
                                        return DropdownMenuItem(
                                          value: pid,
                                          child: Text(
                                            name,
                                            style: const TextStyle(
                                              color: Colors.white,
                                              fontSize: 10,
                                            ),
                                            maxLines: 1,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                        );
                                      }).toList(),
                                      onChanged: (val) {
                                        ref
                                                .read(
                                                  _testTargetPlaceProvider
                                                      .notifier,
                                                )
                                                .state =
                                            val;
                                      },
                                    );
                                  },
                                  loading: () => const Text(
                                    '로딩중...',
                                    style: TextStyle(
                                      color: Colors.white54,
                                      fontSize: 10,
                                    ),
                                  ),
                                  error: (_, __) => const Text(
                                    '에러',
                                    style: TextStyle(
                                      color: Colors.redAccent,
                                      fontSize: 10,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      // ---------- 2열: 캡처 및 이동+캡처 ----------
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: isCapturing
                                  ? null
                                  : () => _triggerCapture(
                                      context,
                                      ref,
                                      'capture',
                                    ),
                              icon: isCapturing
                                  ? const SizedBox(
                                      width: 12,
                                      height: 12,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        color: Colors.white54,
                                      ),
                                    )
                                  : const Icon(
                                      Icons.camera_alt_outlined,
                                      size: 12,
                                    ),
                              label: const Text(
                                '현재캡처',
                                style: TextStyle(fontSize: 10),
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF26293A),
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(
                                  vertical: 12,
                                ),
                                minimumSize: Size.zero,
                              ),
                            ),
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: isCapturing
                                  ? null
                                  : () => _triggerCapture(
                                      context,
                                      ref,
                                      'place_and_capture',
                                    ),
                              icon: isCapturing
                                  ? const SizedBox(
                                      width: 12,
                                      height: 12,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        color: Colors.white54,
                                      ),
                                    )
                                  : const Icon(
                                      Icons.location_on_outlined,
                                      size: 12,
                                    ),
                              label: const Text(
                                '이동+캡처',
                                style: TextStyle(fontSize: 10),
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF1F8CEB),
                                foregroundColor: Colors.white,
                                padding: const EdgeInsets.symmetric(
                                  vertical: 12,
                                ),
                                minimumSize: Size.zero,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  );
                },
              ),

              const SizedBox(height: 10),
            ],
            const SizedBox(height: 10),
            //---------- YOLO 사람 감지 모드 제어 ----------
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'YOLO / AUDIO MODE',
                  style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 11),
                ),
                Consumer(
                  builder: (context, ref, _) {
                    final yoloMode = ref.watch(yoloModeProvider);

                    return Container(
                      height: 38,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        color: const Color(0xFF26293A),
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(color: const Color(0xFF2D3041)),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<int>(
                          value: yoloMode,
                          dropdownColor: const Color(0xFF1C1E2B),
                          icon: const Icon(Icons.arrow_drop_down, color: Colors.white54, size: 16),
                          style: const TextStyle(color: Colors.white, fontSize: 13),
                          items: const [
                            DropdownMenuItem(value: 0, child: Text('OFF', style: TextStyle(fontSize: 12))),
                            DropdownMenuItem(value: 1, child: Text('GLOBAL', style: TextStyle(fontSize: 12))),
                            DropdownMenuItem(value: 2, child: Text('REGION', style: TextStyle(fontSize: 12))),
                          ],
                          onChanged: (val) {
                            if (val != null) {
                              ref.read(yoloModeProvider.notifier).setMode(val);
                            }
                          },
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
            const SizedBox(height: 14),

            //---------- 순찰 루트 및 타임라인 제어 ----------
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      showDialog(
                        context: context,
                        builder: (_) => const PatrolRouteDialog(),
                      );
                    },
                    icon: const Icon(Icons.route_outlined, size: 16),
                    label: const Text(
                      '순찰 루트 설정',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(
                        color: Color(0xFF7F7CFF),
                        width: 1.5,
                      ),
                      foregroundColor: const Color(0xFF7F7CFF),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      showDialog(
                        context: context,
                        builder: (_) => const TimelineDialog(),
                      );
                    },
                    icon: const Icon(Icons.schedule, size: 16),
                    label: const Text(
                      '순찰 타임라인',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF7F7CFF),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Consumer(
                  builder: (context, ref, _) {
                    final isPatrolling = ref.watch(patrolStatusProvider);
                    return _PatrolControlButton(
                      icon: Icons.play_arrow_rounded,
                      label: '순찰 재개',
                      color: isPatrolling
                          ? const Color(0xFF3D4060)
                          : const Color(0xFF4ADE80),
                      onPressed: isPatrolling
                          ? () {}
                          : () => _sendRobotCommand('start_patrol'),
                      isEnabled: !isPatrolling,
                    );
                  },
                ),
                Consumer(
                  builder: (context, ref, _) {
                    final isPatrolling = ref.watch(patrolStatusProvider);
                    return _PatrolControlButton(
                      icon: Icons.pause_rounded,
                      label: '일시 정지',
                      color: isPatrolling
                          ? const Color(0xFFFBBF24)
                          : const Color(0xFF3D4060),
                      onPressed: isPatrolling
                          ? () => _sendRobotCommand('pause_patrol')
                          : () {},
                      isEnabled: isPatrolling,
                    );
                  },
                ),
                Consumer(
                  builder: (context, ref, _) {
                    final isPatrolling = ref.watch(patrolStatusProvider);
                    return _PatrolControlButton(
                      icon: Icons.battery_charging_full_rounded,
                      label: '충전 복귀',
                      color: isPatrolling
                          ? const Color(0xFF38BDF8)
                          : const Color(0xFF3D4060),
                      onPressed: isPatrolling
                          ? () => _sendRobotCommand('return_to_charge')
                          : () {},
                      isEnabled: isPatrolling,
                    );
                  },
                ),
              ],
            ),
            const SizedBox(height: 10),
            //---------- 순찰 진행 현황 (순찰 시작 눌렀을 때만 표시) ----------
            if (ref.watch(patrolStatusProvider)) ...[
              const SizedBox(height: 10),
              const Divider(height: 1, color: Color(0xFF2D3041)),
              const SizedBox(height: 10),

              Consumer(
                builder: (context, ref, _) {
                  final placesAsync = ref.watch(placesProvider);
                  final patrolList = <Map<String, dynamic>>[];
                  placesAsync.whenData((data) {
                    final list = data['places'] as List<dynamic>? ?? [];
                    patrolList.addAll(
                      list
                          .where(
                            (p) =>
                                p['patrol_enabled'] == 1 ||
                                p['patrol_enabled'] == true,
                          )
                          .map((p) => Map<String, dynamic>.from(p)),
                    );
                  });

                  final robotGoal = ref.watch(robotGoalProvider);
                  final currentTargetId = robotGoal?.nextPlaceId;

                  Map<String, dynamic>? currentPlace;
                  if (currentTargetId != null) {
                    try {
                      currentPlace = patrolList.firstWhere((p) => p['place_id'].toString() == currentTargetId);
                    } catch (_) {}
                  }
                  if (currentPlace == null && patrolList.isNotEmpty) {
                    currentPlace = patrolList.first;
                  }

                  final firstId = currentPlace != null ? currentPlace['place_id'].toString() : null;
                  final firstName = currentPlace != null
                      ? (currentPlace['display_name']?.toString() ?? firstId!)
                      : '노드 없음';

                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(
                            Icons.location_on,
                            size: 16,
                            color: Color(0xFF7F7CFF),
                          ),
                          const SizedBox(width: 2),
                          const Text(
                            '순찰 진행중: ',
                            style: TextStyle(
                              color: Color(0xFF9FA4B9),
                              fontSize: 14,
                            ),
                          ),
                          Expanded(
                            child: Text(
                              firstName,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          Consumer(
                            builder: (context, ref, _) {
                              final isPicking = ref.watch(waypointPickingModeProvider);
                              return IconButton(
                                onPressed: _toggleWaypointPickingMode,
                                icon: Icon(
                                  isPicking ? Icons.close : Icons.add_location_alt,
                                  size: 18,
                                ),
                                color: isPicking ? Colors.redAccent : const Color(0xFF7F7CFF),
                                tooltip: isPicking ? '경유점 추가 취소' : '지도에 경유점 추가',
                                padding: EdgeInsets.zero,
                                constraints: const BoxConstraints(),
                              );
                            },
                          ),
                        ],
                      ),
                      if (patrolList.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        SizedBox(
                          height: 30,
                          child: ReorderableListView.builder(
                            scrollDirection: Axis.horizontal,
                            buildDefaultDragHandles: false,
                            itemCount: patrolList.length,
                            onReorder: (oldIndex, newIndex) {
                              if (oldIndex == 0 || newIndex == 0) return; // 현재 노드는 이동 불가
                              if (newIndex > patrolList.length) newIndex = patrolList.length;
                              if (oldIndex < newIndex) newIndex -= 1;
                              
                              final newList = List<Map<String, dynamic>>.from(patrolList);
                              final item = newList.removeAt(oldIndex);
                              newList.insert(newIndex, item);
                              
                              final orderedIds = newList.map((e) => e['place_id'].toString()).toList();
                              ControlActions.reorderPatrol(ref, orderedIds);
                            },
                            proxyDecorator: (child, index, animation) => Material(
                              color: Colors.transparent,
                              child: child,
                            ),
                            itemBuilder: (context, idx) {
                              final p = patrolList[idx];
                              final pid = p['place_id'].toString();
                              final name = p['display_name']?.toString() ?? pid;
                              final isCurrent = pid == firstId;
                              
                              return ReorderableDragStartListener(
                                key: ValueKey(pid),
                                index: idx,
                                enabled: !isCurrent, // 현재 노드는 드래그 비활성화
                                child: Container(
                                  margin: const EdgeInsets.only(right: 8),
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 10,
                                    vertical: 5,
                                  ),
                                  decoration: BoxDecoration(
                                    color: isCurrent
                                        ? const Color(0xFF7F7CFF)
                                        : const Color(0xFF26293A),
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(
                                      color: isCurrent
                                          ? const Color(0xFF7F7CFF)
                                          : const Color(0xFF3D4060),
                                    ),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Text(
                                        '${idx + 1}. $name',
                                        style: TextStyle(
                                          color: isCurrent
                                              ? Colors.white
                                              : const Color(0xFF7F7CFF),
                                          fontSize: 13,
                                          fontWeight: isCurrent
                                              ? FontWeight.w600
                                              : FontWeight.normal,
                                        ),
                                      ),
                                      if (!isCurrent) const SizedBox(width: 4),
                                      if (!isCurrent)
                                        const Icon(Icons.drag_indicator,
                                            size: 14, color: Colors.white24),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      ],
                    ],
                  );
                },
              ),
            ], // if (_isPatrolling) ...[  닫는 ]
          ], // Column children: [  닫는 ]
        ), // Column 닫는 )
      ), // SingleChildScrollView 닫는 )
    ); // Container 닫는 ) 및 return 종결
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

class _PatrolControlButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onPressed;
  final bool isEnabled;

  const _PatrolControlButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onPressed,
    this.isEnabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: isEnabled ? 1.0 : 0.4,
      child: InkWell(
        onTap: isEnabled ? onPressed : null,
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
          child: Column(
            children: [
              Icon(icon, color: color, size: 28),
              const SizedBox(height: 6),
              Text(
                label,
                style: TextStyle(
                  color: isEnabled ? Color(0xFF9FA4B9) : Colors.white38,
                  fontSize: 13, // 11 -> 13으로 확대
                  fontWeight: FontWeight.w400,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
