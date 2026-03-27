import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'dart:math' show pi;
import '../providers/event_provider.dart';
import '../providers/map_provider.dart';
import '../providers/robot_provider.dart';
import '../models/robot_state.dart';
import '../models/event_model.dart';
import '../utils/map_transformer.dart';
import 'event_detail_dialog.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// 맵 상에서 클릭된 이벤트를 추적하기 위한 로컬 상태
final _selectedMapEventProvider = StateProvider<String?>((ref) => null);

class DataCenterMap extends ConsumerWidget {
  const DataCenterMap({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final alerts = ref.watch(eventListProvider);
    final mapImagePathAsync = ref.watch(mapImagePathProvider);
    final mapTransformerAsync = ref.watch(mapTransformerProvider);
    final robotPose = ref.watch(robotPoseProvider);

    //---------- 맵에 표시할 고유한 장소별 최신 이벤트 필터링 ----------
    final Map<String, Event> latestEventsByPlace = {};
    for (var event in alerts) {
      if (!latestEventsByPlace.containsKey(event.placeId)) {
        latestEventsByPlace[event.placeId] = event;
      }
    }

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 맵 본체 프레임 영역 ----------
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xFF10121A),
                borderRadius: BorderRadius.circular(18),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(18),
                child: mapImagePathAsync.when(
                  data: (imagePath) => mapTransformerAsync.when(
                    data: (transformer) => InteractiveViewer(
                      minScale: 1.0,
                      maxScale: 5.0,
                      child: Center(
                        child: FittedBox(
                          fit: BoxFit.contain,
                          alignment: Alignment.center,
                          child: Stack(
                            clipBehavior: Clip.none,
                            children: [
                              //---------- 맵 배경 이미지 렌더링 ----------
                              Image.asset(imagePath),
                              //---------- 각 장소별 최신 이벤트 마커 표시 ----------
                              ...latestEventsByPlace.values.map((event) => _EventMarker(event: event, transformer: transformer)),
                              //---------- 로봇 실시간 마커 렌더링 ----------
                              if (robotPose != null && robotPose.x != null && robotPose.y != null)
                                _RobotMarker(pose: robotPose, transformer: transformer),
                            ],
                          ),
                        ),
                      ),
                    ),
                    loading: () => const Center(child: CircularProgressIndicator(color: Color(0xFF38BDF8))),
                    error: (err, stack) => Center(child: Text('Map Transform Error: $err', style: const TextStyle(color: Colors.red))),
                  ),
                  loading: () => const Center(child: CircularProgressIndicator(color: Color(0xFF38BDF8))),
                  error: (err, stack) => Center(child: Text('Map Image Error: $err', style: const TextStyle(color: Colors.red))),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _RobotMarker extends StatelessWidget {
  final RobotPose pose;
  final MapTransformer transformer;

  const _RobotMarker({required this.pose, required this.transformer});

  @override
  Widget build(BuildContext context) {
    if (pose.x == null || pose.y == null) return const SizedBox();
    
    final transformed = transformer.transform(pose.x!, pose.y!, pose.yaw ?? 0);
    final px = transformed['px']!;
    final py = transformed['py']!;
    final guiYaw = transformed['yaw']!;

    return Positioned(
      left: px - 20, // size 40 / 2
      top: py - 20,
      child: Transform.rotate(
        angle: guiYaw + (pi / 2),
        child: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: const Color(0xFF1F8CEB).withOpacity(0.3),
            shape: BoxShape.circle,
            border: Border.all(color: const Color(0xFF1F8CEB), width: 2),
            boxShadow: const [
              BoxShadow(color: Color(0x661F8CEB), blurRadius: 10, spreadRadius: 2)
            ],
          ),
          child: const Center(
            child: Icon(Icons.navigation_rounded, color: Colors.white, size: 24),
          ),
        ),
      ),
    );
  }
}

class _EventMarker extends ConsumerWidget {
  final Event event;
  final MapTransformer transformer;
  const _EventMarker({required this.event, required this.transformer});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAnomaly = event.anomalyFlag == 1;
    final selectedId = ref.watch(_selectedMapEventProvider);
    final isSelected = selectedId == event.eventId;

    //---------- 실제 DB 좌표 기반 맵 픽셀 매핑 ----------
    double px = 0;
    double py = 0;

    if (event.x != null && event.y != null) {
      final transformed = transformer.transform(event.x!, event.y!, event.yaw ?? 0);
      px = transformed['px']!;
      py = transformed['py']!;
    } else {
      // Fallback: 좌표가 없는 예전 데이터들의 경우 해시를 이용한 임의 배치
      final hash = event.placeId.hashCode;
      px = (hash % 1000) + 100.0;
      py = ((hash ~/ 1000) % 1000) + 100.0;
    }

    // 마커 박스의 전체 가로세로 크기 고정
    const markerWidth = 220.0;
    const markerHeight = 320.0;

    return Positioned(
      left: px - (markerWidth / 2),
      top: py - (markerHeight / 2),
      child: SizedBox(
        width: markerWidth, 
        height: markerHeight,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center, // 마커(점)를 정중앙 배치
          children: [
            // 말풍선 (선택 시 마커 상단에 표시)
            if (isSelected)
              Positioned(
                bottom: (markerHeight / 2) + 15,
                child: GestureDetector(
                  onTap: () => showEventDetailDialog(context, ref, event),
                  child: Container(
                    width: 210,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1E2B),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isAnomaly ? const Color(0xFFEF4444) : const Color(0xFF38BDF8),
                        width: 1.5,
                      ),
                      boxShadow: const [
                        BoxShadow(color: Colors.black54, blurRadius: 15, offset: Offset(0, 8)),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(
                              child: Text(
                                event.summaryText ?? '이벤트 발생',
                                style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            const Icon(Icons.open_in_new, color: Colors.white54, size: 14),
                          ],
                        ),
                        const SizedBox(height: 8),
                        // 이미지 영역
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Container(
                            height: 110,
                            width: double.infinity,
                            decoration: const BoxDecoration(color: Color(0xFF11121A)),
                            child: event.frames.isNotEmpty
                                ? Image.network(
                                    'http://localhost:8000/images/${event.frames.first.imagePath.replaceFirst("recv/", "")}',
                                    fit: BoxFit.cover,
                                    errorBuilder: (_, __, ___) => const Icon(Icons.broken_image, color: Colors.grey, size: 24),
                                  )
                                : const Icon(Icons.image_not_supported, color: Colors.grey, size: 24),
                          ),
                        ),
                        const SizedBox(height: 6),
                        const Center(
                          child: Text(
                            '클릭하여 상세 정보 확인',
                            style: TextStyle(color: Colors.white38, fontSize: 10, fontStyle: FontStyle.italic),
                          ),
                        ),
                        const SizedBox(height: 6),
                        // 모드 변경
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton(
                            style: OutlinedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 4),
                              side: const BorderSide(color: Color(0xFF393C4B)),
                            ),
                            onPressed: () async {
                              showMenu(
                                context: context,
                                position: const RelativeRect.fromLTRB(100, 100, 0, 0),
                                color: const Color(0xFF1C1E2B),
                                items: const [
                                  PopupMenuItem(value: 'idle', child: Text('idle', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'bank', child: Text('bank', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'th_calib', child: Text('th_calib', style: TextStyle(color: Colors.white))),
                                  PopupMenuItem(value: 'query', child: Text('query', style: TextStyle(color: Colors.white))),
                                ],
                              ).then((value) {
                                if (value != null) {
                                  http.post(
                                    Uri.parse('http://127.0.0.1:8000/places/${event.placeId}/config'),
                                    body: {'mode': value},
                                  ).then((_) {
                                    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('${event.placeId} 구역 모드 변경: $value')));
                                  });
                                }
                              });
                            },
                            child: const Text('운영 모드 변경 (m)', style: TextStyle(fontSize: 10, color: Color(0xFFB5BAD3))),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            
            // 마커 본체
            GestureDetector(
              onTap: () {
                final notifier = ref.read(_selectedMapEventProvider.notifier);
                notifier.state = (notifier.state == event.eventId) ? null : event.eventId;
              },
              child: isAnomaly
                  ? const _PulsingDot(color: Color(0xFFFF4B5C), size: 24)
                  : Container(
                      width: 16,
                      height: 16,
                      decoration: BoxDecoration(
                        color: const Color(0xFF38BDF8),
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 2),
                        boxShadow: const [
                          BoxShadow(color: Color(0x6638BDF8), blurRadius: 10, spreadRadius: 2)
                        ],
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

// 비정상 마커용 펄스(Ripple) 애니메이션 마커
class _PulsingDot extends StatefulWidget {
  final Color color;
  final double size;
  const _PulsingDot({required this.color, required this.size});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 2))..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: Stack(
        alignment: Alignment.center,
        clipBehavior: Clip.none,
        children: [
          AnimatedBuilder(
            animation: _controller,
            builder: (context, child) {
              return Opacity(
                opacity: 1.0 - _controller.value,
                child: Container(
                  width: widget.size + (widget.size * 2 * _controller.value),
                  height: widget.size + (widget.size * 2 * _controller.value),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: widget.color.withOpacity(0.5),
                  ),
                ),
              );
            },
          ),
          Container(
            width: widget.size,
            height: widget.size,
            decoration: BoxDecoration(shape: BoxShape.circle, color: widget.color),
            child: const Icon(Icons.warning_rounded, size: 16, color: Colors.white),
          ),
        ],
      ),
    );
  }
}
