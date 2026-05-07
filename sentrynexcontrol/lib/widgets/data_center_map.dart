import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'dart:math' show pi;
import 'dart:convert';
import '../providers/event_provider.dart';
import '../providers/map_provider.dart';
import '../providers/robot_provider.dart';
import '../features/control/control_provider.dart';
import '../providers/audio_provider.dart';
import '../providers/yolo_regions_provider.dart';
import '../models/robot_state.dart';
import '../models/event_model.dart';
import '../utils/map_transformer.dart';
import 'event_detail_dialog.dart';
import '../providers/auth_event_provider.dart';
import '../models/auth_event_model.dart';
import 'package:http/http.dart' as http;

final _selectedMapEventProvider = StateProvider<String?>((ref) => null);

class DataCenterMap extends ConsumerStatefulWidget {
  const DataCenterMap({super.key});

  @override
  ConsumerState<DataCenterMap> createState() => _DataCenterMapState();
}

class _DataCenterMapState extends ConsumerState<DataCenterMap> {
  Offset? _dragStart;
  Offset? _dragEnd;
  final TransformationController _transformationController =
      TransformationController();
  bool _isTrackingMode = false;
  Size _viewportSize = Size.zero;

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  void _promptSaveRegion(double xMin, double xMax, double yMin, double yMax) {
    if ((xMin - xMax).abs() < 0.2 || (yMin - yMax).abs() < 0.2) return;

    final nameController = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1C1E2B),
        title: const Text('YOLO 구역 추가', style: TextStyle(color: Colors.white)),
        content: TextField(
          controller: nameController,
          style: const TextStyle(color: Colors.white),
          decoration: const InputDecoration(
            hintText: '구역 이름 (예: 보안 구역 A)',
            hintStyle: TextStyle(color: Colors.white54),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              setState(() {
                _dragStart = null;
                _dragEnd = null;
              });
            },
            child: const Text('취소', style: TextStyle(color: Colors.white54)),
          ),
          ElevatedButton(
            onPressed: () {
              if (nameController.text.trim().isNotEmpty) {
                ref
                    .read(yoloRegionsProvider.notifier)
                    .addRegion(
                      nameController.text.trim(),
                      xMin,
                      xMax,
                      yMin,
                      yMax,
                    );
              }
              Navigator.pop(ctx);
              setState(() {
                _dragStart = null;
                _dragEnd = null;
                ref.read(yoloDrawingModeProvider.notifier).state = false;
              });
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF7F7CFF),
            ),
            child: const Text('추가', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Future<void> _promptSaveWaypoint(double x, double y) async {
    final nameController = TextEditingController(text: '경유점');
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1C1E2B),
        title: const Text('경유점 추가', style: TextStyle(color: Colors.white)),
        content: TextField(
          controller: nameController,
          style: const TextStyle(color: Colors.white),
          decoration: const InputDecoration(
            labelText: '경유점 이름',
            labelStyle: TextStyle(color: Colors.white54),
            enabledBorder: UnderlineInputBorder(borderSide: BorderSide(color: Colors.white24)),
            focusedBorder: UnderlineInputBorder(borderSide: BorderSide(color: Color(0xFF7F7CFF))),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, null),
            child: const Text('취소', style: TextStyle(color: Colors.white54)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF7F7CFF)),
            onPressed: () => Navigator.pop(ctx, nameController.text),
            child: const Text('추가', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );

    if (result == null || result.isEmpty) {
      ref.read(waypointPickingModeProvider.notifier).state = false;
      return;
    }

    try {
      final res = await http.post(
        Uri.parse('http://127.0.0.1:8000/gui/waypoints'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'x': x,
          'y': y,
          'yaw': 0.0,
          'display_name': result,
        }),
      );

      if (res.statusCode == 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('경유점 "$result"이(가) 추가되었습니다.')),
          );
        }
        ref.invalidate(placesProvider);
        if (ref.read(patrolStatusProvider)) {
          await http.post(
            Uri.parse('http://127.0.0.1:8000/robot/command'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'command': 'start_patrol'}),
          );
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('경유점 추가 실패: ${res.statusCode}')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('서버 통신 오류: $e')),
        );
      }
    } finally {
      ref.read(waypointPickingModeProvider.notifier).state = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    final alerts = ref.watch(eventListProvider);
    final placesAsync = ref.watch(placesProvider);
    final audioEvents = ref.watch(audioEventListProvider);
    final yoloRegionsAsync = ref.watch(yoloRegionsProvider);
    final mapImagePathAsync = ref.watch(mapImagePathProvider);
    final mapTransformerAsync = ref.watch(mapTransformerProvider);
    final robotPose = ref.watch(robotPoseProvider);
    final authEvents = ref.watch(authEventListProvider);
    final isDrawingMode = ref.watch(yoloDrawingModeProvider);
    final isWaypointPicking = ref.watch(waypointPickingModeProvider);

    ref.listen(robotPoseProvider, (previous, next) {
      if (_isTrackingMode && next != null && next.x != null && next.y != null && mapTransformerAsync.hasValue) {
        final transformer = mapTransformerAsync.value!;
        final transformed = transformer.transform(next.x!, next.y!, 0);
        
        if (_viewportSize.width > 0 && _viewportSize.height > 0) {
          final scale = _viewportSize.width / transformer.imageWidth;
          final pyScaled = transformed['py']! * scale;
          
          final childHeight = transformer.imageHeight * scale;
          double clampedTx = 0;
          double clampedTy = (_viewportSize.height / 2) - pyScaled;
          
          if (childHeight > _viewportSize.height) {
            final minTy = _viewportSize.height - childHeight;
            clampedTy = clampedTy.clamp(minTy, 0.0);
          } else {
            clampedTy = (_viewportSize.height - childHeight) / 2;
          }
          
          _transformationController.value = Matrix4.identity()..translate(clampedTx, clampedTy);
        }
      }
    });

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
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '데이터센터 맵',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Consumer(
                builder: (context, ref, _) {
                  final isDrawing = ref.watch(yoloDrawingModeProvider);
                  final showRegions = ref.watch(yoloShowRegionsProvider);

                  return Row(
                    children: [
                      // 구역 표시 토글 (체크박스 스타일 대신 Switch/IconButton 조합)
                      const Text(
                        '보안구역 표시',
                        style: TextStyle(color: Colors.white54, fontSize: 12),
                      ),
                      Transform.scale(
                        scale: 0.7,
                        child: Switch(
                          value: showRegions,
                          onChanged: (val) =>
                              ref.read(yoloShowRegionsProvider.notifier).state =
                                  val,
                          activeColor: const Color(0xFF4ADE80),
                        ),
                      ),
                      const SizedBox(width: 8),
                      ElevatedButton.icon(
                        onPressed: () {
                          ref.read(yoloDrawingModeProvider.notifier).state =
                              !isDrawing;
                        },
                        icon: Icon(
                          isDrawing ? Icons.close : Icons.add,
                          size: 14,
                        ),
                        label: Text(
                          isDrawing ? '추가 중단' : '구역 추가',
                          style: const TextStyle(fontSize: 12),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: isDrawing
                              ? const Color(0xFF3D4060)
                              : const Color(0xFF181924),
                          foregroundColor: Colors.white70,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 12,
                          ),
                          minimumSize: Size.zero,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(30),
                            side: const BorderSide(color: Color(0xFF3D4060)),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ],
          ),
          const SizedBox(height: 12),
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xFF10121A),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: mapImagePathAsync.when(
                      data: (imagePath) => mapTransformerAsync.when(
                        data: (transformer) {
                          final robotGoal = ref.watch(robotGoalProvider);
                          // 목표 지점 찾기
                          Map<String, dynamic>? targetPlace;
                          if (robotGoal?.nextPlaceId != null &&
                              placesAsync.hasValue) {
                            final places =
                                placesAsync.value?['places'] as List<dynamic>?;
                            if (places != null) {
                              for (var p in places) {
                                if (p['place_id'].toString() ==
                                    robotGoal!.nextPlaceId) {
                                  targetPlace = p;
                                  break;
                                }
                              }
                            }
                          }
                          return LayoutBuilder(
                            builder: (context, constraints) {
                              // Constraints 사이즈를 렌더링 사이클 이후에 _viewportSize에 저장
                              WidgetsBinding.instance.addPostFrameCallback((_) {
                                if (mounted) {
                                  _viewportSize = Size(constraints.maxWidth, constraints.maxHeight);
                                }
                              });
                              return InteractiveViewer(
                                constrained: false,
                                transformationController: _transformationController,
                                panEnabled: !isDrawingMode && !isWaypointPicking,
                                scaleEnabled: !isDrawingMode && !isWaypointPicking,
                                minScale: 1.0,
                                maxScale: 5.0,
                                boundaryMargin: EdgeInsets.zero,
                                child: SizedBox(
                                  width: constraints.maxWidth,
                                  height: _isTrackingMode
                                      ? constraints.maxWidth * (transformer.imageHeight / transformer.imageWidth)
                                      : constraints.maxHeight,
                                  child: FittedBox(
                                    fit: _isTrackingMode ? BoxFit.fill : BoxFit.contain,
                                    child: SizedBox(
                                      width: transformer.imageWidth,
                                      height: transformer.imageHeight,
                                      child: GestureDetector(
                                  onPanStart: isDrawingMode
                                      ? (details) {
                                          setState(() {
                                            _dragStart = details.localPosition;
                                            _dragEnd = details.localPosition;
                                          });
                                        }
                                      : null,
                                  onTapDown: isWaypointPicking
                                      ? (details) async {
                                          final r = transformer.inverseTransform(
                                            details.localPosition.dx,
                                            details.localPosition.dy,
                                            0,
                                          );
                                          _promptSaveWaypoint(r['x']!, r['y']!);
                                        }
                                      : null,
                                  onPanUpdate: isDrawingMode
                                      ? (details) {
                                          setState(() {
                                            _dragEnd = details.localPosition;
                                          });
                                        }
                                      : null,
                                  onPanEnd: isDrawingMode
                                      ? (details) {
                                          if (_dragStart != null &&
                                              _dragEnd != null) {
                                            final rStart = transformer
                                                .inverseTransform(
                                                  _dragStart!.dx,
                                                  _dragStart!.dy,
                                                  0,
                                                );
                                            final rEnd = transformer
                                                .inverseTransform(
                                                  _dragEnd!.dx,
                                                  _dragEnd!.dy,
                                                  0,
                                                );
                                            _promptSaveRegion(
                                              rStart['x']!,
                                              rEnd['x']!,
                                              rStart['y']!,
                                              rEnd['y']!,
                                            );
                                          }
                                        }
                                      : null,
                                  child: Stack(
                                    clipBehavior: Clip.none,
                                    children: [
                                      Image.asset(imagePath),

                                      // 경유점 추가 모드 오버레이
                                      if (isWaypointPicking)
                                        Positioned.fill(
                                          child: Container(
                                            color: Colors.black.withOpacity(0.5),
                                            alignment: Alignment.center,
                                            child: const Text(
                                              '원하는 위치를 클릭하여 경유점을 추가하세요',
                                              style: TextStyle(
                                                color: Colors.white,
                                                fontSize: 16,
                                                fontWeight: FontWeight.bold,
                                                shadows: [Shadow(color: Colors.black, blurRadius: 4)],
                                              ),
                                            ),
                                          ),
                                        ),

                                      if (ref.watch(yoloShowRegionsProvider) &&
                                          yoloRegionsAsync.value != null)
                                        ...yoloRegionsAsync.value!.map(
                                          (r) => _YoloRegionMarker(
                                            region: r,
                                            transformer: transformer,
                                          ),
                                        ),
                                      if (placesAsync.hasValue &&
                                          placesAsync.value?['places'] != null)
                                        ...(placesAsync.value!['places']
                                                as List<dynamic>)
                                            .map(
                                              (p) => _PlaceMarker(
                                                place: p,
                                                latestEvent:
                                                    latestEventsByPlace[p['place_id']
                                                        .toString()],
                                                transformer: transformer,
                                              ),
                                            ),

                                      if (authEvents.isNotEmpty)
                                        ...authEvents.map(
                                          (e) => _AuthMarker(
                                            event: e,
                                            transformer: transformer,
                                          ),
                                        ),

                                      if (robotPose != null &&
                                          robotPose.x != null &&
                                          robotPose.y != null) ...[
                                        if (targetPlace != null)
                                          _TargetLinePainterWidget(
                                            robotPose: robotPose,
                                            targetPlace: targetPlace,
                                            transformer: transformer,
                                          ),
                                        _RobotMarker(
                                          pose: robotPose,
                                          transformer: transformer,
                                        ),
                                      ],

                                      ...audioEvents
                                          .where(
                                            (e) =>
                                                e.adminChecked == 0 &&
                                                e.x != null &&
                                                e.y != null,
                                          )
                                          .map((audio) {
                                            return _AudioMarker(
                                              audio: audio,
                                              transformer: transformer,
                                            );
                                          }),

                                      if (isDrawingMode &&
                                          _dragStart != null &&
                                          _dragEnd != null)
                                        Positioned(
                                          left: _dragStart!.dx < _dragEnd!.dx
                                              ? _dragStart!.dx
                                              : _dragEnd!.dx,
                                          top: _dragStart!.dy < _dragEnd!.dy
                                              ? _dragStart!.dy
                                              : _dragEnd!.dy,
                                          width: (_dragEnd!.dx - _dragStart!.dx)
                                              .abs(),
                                          height:
                                              (_dragEnd!.dy - _dragStart!.dy)
                                                  .abs(),
                                          child: Container(
                                            decoration: BoxDecoration(
                                              border: Border.all(
                                                color: const Color(0xFF4ADE80),
                                                width: 2,
                                              ),
                                              color: const Color(
                                                0xFF4ADE80,
                                              ).withOpacity(0.2),
                                            ),
                                          ),
                                        ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                            ),
                          );
                        },
                      );
                    },
                    loading: () => const Center(
                          child: CircularProgressIndicator(
                            color: Color(0xFF38BDF8),
                          ),
                        ),
                        error: (err, stack) => Center(
                          child: Text(
                            'Map Transform Error: $err',
                            style: const TextStyle(color: Colors.red),
                          ),
                        ),
                      ),
                      loading: () => const Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFF38BDF8),
                        ),
                      ),
                      error: (err, stack) => Center(
                        child: Text(
                          'Map Image Error: $err',
                          style: const TextStyle(color: Colors.red),
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    right: 20,
                    bottom: 20,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Tooltip(
                          message: _isTrackingMode ? '로봇 추적 끄기' : '로봇 추적 켜기',
                          child: FloatingActionButton(
                            mini: true,
                            backgroundColor: _isTrackingMode ? const Color(0xFF7F7CFF) : const Color(0xFF1C1E2B),
                            foregroundColor: Colors.white70,
                            onPressed: () {
                              setState(() {
                                _isTrackingMode = !_isTrackingMode;
                              });
                            },
                            child: Icon(
                              _isTrackingMode ? Icons.my_location : Icons.location_searching,
                              size: 20,
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Tooltip(
                          message: '전체 화면',
                          child: FloatingActionButton(
                            mini: true,
                            backgroundColor: const Color(0xFF1C1E2B),
                            foregroundColor: Colors.white70,
                            onPressed: () {
                              _transformationController.value = Matrix4.identity();
                              setState(() {
                                _isTrackingMode = false;
                              });
                            },
                            child: const Icon(Icons.zoom_out_map, size: 20),
                          ),
                        ),
                      ],
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

    // 마커 크기를 늘려 시야각(FOV) 부채꼴을 그릴 수 있게 함
    const double markerSize = 120.0;

    return Positioned(
      left: px - (markerSize / 2),
      top: py - (markerSize / 2),
      child: Transform.rotate(
        angle: guiYaw + (pi / 2),
        child: SizedBox(
          width: markerSize,
          height: markerSize,
          child: CustomPaint(painter: _RobotPainter()),
        ),
      ),
    );
  }
}

class _RobotPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final fovRadius = size.width / 2;

    // 1. 시야각(FOV) 위쪽 방향으로 선명하게 그리기
    final fovPaint = Paint()
      ..shader = RadialGradient(
        colors: [
          const Color(0xFF4ADE80).withOpacity(0.6),
          const Color(0xFF4ADE80).withOpacity(0.15), // 끝부분도 어느정도 보이게
        ],
        stops: const [0.2, 1.0],
      ).createShader(Rect.fromCircle(center: center, radius: fovRadius));

    final startAngle = -pi / 2 - pi / 4;
    final sweepAngle = pi / 2;

    // FOV 내부 채우기 (기준 위쪽 -90도, 90도 시야각)
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: fovRadius),
      startAngle,
      sweepAngle,
      true,
      fovPaint,
    );

    // FOV 외곽선 (양옆 경계선을 선명하게)
    final fovBorderPaint = Paint()
      ..color = const Color(0xFF4ADE80).withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: fovRadius),
      startAngle,
      sweepAngle,
      true,
      fovBorderPaint,
    );

    // 2. 로봇 바퀴 (약한 그레이, 뒤쪽 배치)
    final robotWidth = 20.0;
    final robotHeight = 24.0;
    final wheelWidth = 6.0;
    final wheelHeight = 10.0;
    final wheelPaint = Paint()..color = const Color(0xFF757B92);

    // 전면이 위쪽이므로 뒤쪽은 y가 중심에서 살짝 아래(+y)
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: center + Offset(-robotWidth / 2, robotHeight / 4 + 2),
          width: wheelWidth,
          height: wheelHeight,
        ),
        const Radius.circular(2),
      ),
      wheelPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: center + Offset(robotWidth / 2, robotHeight / 4 + 2),
          width: wheelWidth,
          height: wheelHeight,
        ),
        const Radius.circular(2),
      ),
      wheelPaint,
    );

    // 3. 로봇 본체 (밝은 그레이 둥근 직사각형)
    final robotRect = Rect.fromCenter(
      center: center,
      width: robotWidth,
      height: robotHeight,
    );
    final glowPaint = Paint()
      ..color = const Color(0xFFE2E8F0).withOpacity(0.3)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
    canvas.drawRRect(
      RRect.fromRectAndRadius(robotRect, const Radius.circular(6)),
      glowPaint,
    );

    final bodyPaint = Paint()
      ..color =
          const Color(0xFFF1F5F9) // 아주 밝은 그레이
      ..style = PaintingStyle.fill;
    final borderPaint = Paint()
      ..color =
          const Color(0xFF94A3B8) // 약간 진한 윤곽선
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    canvas.drawRRect(
      RRect.fromRectAndRadius(robotRect, const Radius.circular(6)),
      bodyPaint,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(robotRect, const Radius.circular(6)),
      borderPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _PlaceMarker extends ConsumerWidget {
  final dynamic place;
  final Event? latestEvent;
  final MapTransformer transformer;

  const _PlaceMarker({
    required this.place,
    required this.latestEvent,
    required this.transformer,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAnomaly = latestEvent?.anomalyFlag == 1;
    final isPatrolEnabled =
        place['patrol_enabled'] == 1 ||
        place['patrol_enabled'] == true ||
        place['patrol_enabled'] == '1' ||
        place['patrol_enabled'] == 'true';
    final placeId = place['place_id'].toString();
    final displayName = place['display_name']?.toString() ?? placeId;

    final selectedId = ref.watch(_selectedMapEventProvider);
    final isSelected = selectedId == placeId;

    double px = 0;
    double py = 0;

    if (place['x'] != null && place['y'] != null) {
      final transformed = transformer.transform(
        place['x'],
        place['y'],
        place['yaw'] ?? 0,
      );
      px = transformed['px']!;
      py = transformed['py']!;
    } else {
      final hash = placeId.hashCode;
      px = (hash % 1000) + 100.0;
      py = ((hash ~/ 1000) % 1000) + 100.0;
    }

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
          alignment: Alignment.center,
          children: [
            if (isSelected)
              Positioned(
                bottom: (markerHeight / 2) + 15,
                child: GestureDetector(
                  onTap: () {
                    if (latestEvent != null) {
                      showEventDetailDialog(context, ref, latestEvent!);
                    } else {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('\'$displayName\'에서의 최신 이벤트 기록이 없습니다.'),
                        ),
                      );
                    }
                  },
                  child: Container(
                    width: 210,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1E2B),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isAnomaly
                            ? const Color(0xFFEF4444)
                            : const Color(0xFF38BDF8),
                        width: 1.5,
                      ),
                      boxShadow: const [
                        BoxShadow(
                          color: Colors.black54,
                          blurRadius: 15,
                          offset: Offset(0, 8),
                        ),
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
                                latestEvent?.summaryText ?? '기록된 이벤트 없음',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            if (latestEvent != null)
                              const Icon(
                                Icons.open_in_new,
                                color: Colors.white54,
                                size: 14,
                              ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        if (latestEvent != null)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: Container(
                              height: 110,
                              width: double.infinity,
                              decoration: const BoxDecoration(
                                color: Color(0xFF11121A),
                              ),
                              child: latestEvent!.frames.isNotEmpty
                                  ? Image.network(
                                      'http://localhost:8000/images/${latestEvent!.frames.first.imagePath.replaceFirst("recv/", "")}',
                                      fit: BoxFit.cover,
                                      errorBuilder: (_, __, ___) => const Icon(
                                        Icons.broken_image,
                                        color: Colors.grey,
                                        size: 24,
                                      ),
                                    )
                                  : const Icon(
                                      Icons.image_not_supported,
                                      color: Colors.grey,
                                      size: 24,
                                    ),
                            ),
                          ),
                        const SizedBox(height: 6),
                        const Center(
                          child: Text(
                            '클릭하여 정보 확인',
                            style: TextStyle(
                              color: Colors.white38,
                              fontSize: 10,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                        const SizedBox(height: 6),
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
                                position: const RelativeRect.fromLTRB(
                                  100,
                                  100,
                                  0,
                                  0,
                                ),
                                color: const Color(0xFF1C1E2B),
                                items: const [
                                  PopupMenuItem(
                                    value: 'idle',
                                    child: Text(
                                      'idle',
                                      style: TextStyle(color: Colors.white),
                                    ),
                                  ),
                                  PopupMenuItem(
                                    value: 'bank',
                                    child: Text(
                                      'bank',
                                      style: TextStyle(color: Colors.white),
                                    ),
                                  ),
                                  PopupMenuItem(
                                    value: 'th_calib',
                                    child: Text(
                                      'th_calib',
                                      style: TextStyle(color: Colors.white),
                                    ),
                                  ),
                                  PopupMenuItem(
                                    value: 'query',
                                    child: Text(
                                      'query',
                                      style: TextStyle(color: Colors.white),
                                    ),
                                  ),
                                ],
                              ).then((value) {
                                if (value != null) {
                                  http
                                      .post(
                                        Uri.parse(
                                          'http://127.0.0.1:8000/places/$placeId/config',
                                        ),
                                        body: {'mode': value},
                                      )
                                      .then((_) {
                                        if (context.mounted)
                                          ScaffoldMessenger.of(
                                            context,
                                          ).showSnackBar(
                                            SnackBar(
                                              content: Text(
                                                '$placeId 구역 모드 변경: $value',
                                              ),
                                            ),
                                          );
                                      });
                                }
                              });
                            },
                            child: const Text(
                              '운영 모드 변경 (m)',
                              style: TextStyle(
                                fontSize: 10,
                                color: Color(0xFFB5BAD3),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

            GestureDetector(
              onTap: () {
                final notifier = ref.read(_selectedMapEventProvider.notifier);
                notifier.state = (notifier.state == placeId) ? null : placeId;
              },
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  isAnomaly
                      ? const _PulsingDot(color: Color(0xFFFF4B5C), size: 24)
                      : (place['place_type'] == 'waypoint'
                          ? Icon(
                              Icons.place,
                              color: isPatrolEnabled
                                  ? const Color(0xFFFACC15) // 경유점은 노란색 핀
                                  : const Color(0xFF6B7280),
                              size: 24,
                            )
                          : Container(
                              width: 16,
                              height: 16,
                              decoration: BoxDecoration(
                                color: isPatrolEnabled
                                    ? const Color(0xFF38BDF8) // 캡처 지점은 파란색 원
                                    : const Color(0xFF6B7280),
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white, width: 2),
                                boxShadow: [
                                  if (isPatrolEnabled)
                                    const BoxShadow(
                                      color: Color(0xCC38BDF8),
                                      blurRadius: 15,
                                      spreadRadius: 3,
                                    ),
                                ],
                              ),
                            )),
                  const SizedBox(height: 4),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 6,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.transparent,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      displayName,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PulsingDot extends StatefulWidget {
  final Color color;
  final double size;
  final IconData? icon;
  const _PulsingDot({required this.color, required this.size, this.icon = Icons.warning_rounded});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
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
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: widget.color,
            ),
            child: Icon(
              widget.icon,
              size: 16,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

class _AudioMarker extends ConsumerWidget {
  final dynamic audio;
  final MapTransformer transformer;
  const _AudioMarker({required this.audio, required this.transformer});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (audio.x == null || audio.y == null) return const SizedBox.shrink();

    final transformed = transformer.transform(
      audio.x!,
      audio.y!,
      audio.yaw ?? 0,
    );
    final px = transformed['px']!;
    final py = transformed['py']!;

    return Positioned(
      left: px - 12,
      top: py - 12,
      child: Stack(
        clipBehavior: Clip.none,
        alignment: Alignment.center,
        children: [
          const _PulsingDot(color: Color(0xFFBA68C8), size: 18, icon: Icons.volume_up),
        ],
      ),
    );
  }
}

class _YoloRegionMarker extends ConsumerWidget {
  final Map<String, dynamic> region;
  final MapTransformer transformer;

  const _YoloRegionMarker({required this.region, required this.transformer});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final px1 = transformer.transform(
      region['x_min'] * 1.0,
      region['y_min'] * 1.0,
      0,
    )['px']!;
    final py1 = transformer.transform(
      region['x_min'] * 1.0,
      region['y_min'] * 1.0,
      0,
    )['py']!;
    final px2 = transformer.transform(
      region['x_max'] * 1.0,
      region['y_max'] * 1.0,
      0,
    )['px']!;
    final py2 = transformer.transform(
      region['x_max'] * 1.0,
      region['y_max'] * 1.0,
      0,
    )['py']!;

    final double left = px1 < px2 ? px1 : px2;
    final double top = py1 < py2 ? py1 : py2;
    final double width = (px1 - px2).abs();
    final double height = (py1 - py2).abs();

    final bool isEnabled =
        region['is_enabled'] == 1 || region['is_enabled'] == true;

    return Positioned(
      left: left,
      top: top,
      width: width,
      height: height,
      child: GestureDetector(
        onTap: () {
          showDialog(
            context: context,
            builder: (ctx) => AlertDialog(
              backgroundColor: const Color(0xFF1C1E2B),
              title: Text(
                '${region['name']}',
                style: const TextStyle(color: Colors.white),
              ),
              content: Text(
                '현재 상태 : ${isEnabled ? "ON" : "OFF"}',
                style: const TextStyle(color: Colors.white70),
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(ctx);
                    ref
                        .read(yoloRegionsProvider.notifier)
                        .toggleRegionEnabled(region['region_id'], !isEnabled);
                  },
                  child: Text(
                    isEnabled ? '비활성화 하기' : '활성화 하기',
                    style: const TextStyle(color: Color(0xFFFBBF24)),
                  ),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pop(ctx);
                    ref
                        .read(yoloRegionsProvider.notifier)
                        .deleteRegion(region['region_id']);
                  },
                  child: const Text(
                    '삭제',
                    style: TextStyle(color: Color(0xFFEF4444)),
                  ),
                ),
                ElevatedButton(
                  onPressed: () => Navigator.pop(ctx),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF7F7CFF),
                  ),
                  child: const Text(
                    '닫기',
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ],
            ),
          );
        },
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(
              color: isEnabled
                  ? const Color(0xFF4ADE80)
                  : const Color(0xFF6B7280),
              width: 1.5,
            ),
            color: isEnabled
                ? const Color(0xFF4ADE80).withOpacity(0.05)
                : const Color(0xFF6B7280).withOpacity(0.05),
          ),
          child: Center(
            child: Text(
              region['name'] ?? '구역',
              style: TextStyle(
                color: isEnabled
                    ? const Color(0xFF4ADE80)
                    : const Color(0xFF6B7280),
                fontSize: 10,
                fontWeight: FontWeight.bold,
                shadows: const [Shadow(color: Colors.black87, blurRadius: 4)],
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );
  }
}

class _TargetLinePainterWidget extends StatefulWidget {
  final dynamic robotPose;
  final Map<String, dynamic> targetPlace;
  final MapTransformer transformer;

  const _TargetLinePainterWidget({
    required this.robotPose,
    required this.targetPlace,
    required this.transformer,
  });

  @override
  State<_TargetLinePainterWidget> createState() =>
      _TargetLinePainterWidgetState();
}

class _TargetLinePainterWidgetState extends State<_TargetLinePainterWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return CustomPaint(
          painter: _DynamicLinePainter(
            robotX: widget.robotPose.x!,
            robotY: widget.robotPose.y!,
            targetX: double.tryParse(widget.targetPlace['x'].toString()) ?? 0,
            targetY: double.tryParse(widget.targetPlace['y'].toString()) ?? 0,
            transformer: widget.transformer,
            pulseValue: _controller.value,
          ),
        );
      },
    );
  }
}

class _DynamicLinePainter extends CustomPainter {
  final double robotX;
  final double robotY;
  final double targetX;
  final double targetY;
  final MapTransformer transformer;
  final double pulseValue;

  _DynamicLinePainter({
    required this.robotX,
    required this.robotY,
    required this.targetX,
    required this.targetY,
    required this.transformer,
    required this.pulseValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final robotPos = transformer.transform(robotX, robotY, 0);
    final targetPos = transformer.transform(targetX, targetY, 0);

    final start = Offset(robotPos['px']!, robotPos['py']!);
    final end = Offset(targetPos['px']!, targetPos['py']!);

    // 1. 점선(Dashed Line) 그리기
    final paint = Paint()
      ..color = const Color(0xFF7F7CFF).withOpacity(0.9) // 불투명도 상향
      ..strokeWidth = 2.0 // 선 굵기 상향
      ..style = PaintingStyle.stroke;

    _drawDashedLine(canvas, start, end, paint);

    // 2. 타겟 노드 펄스(Pulse) 효과
    final pulsePaint = Paint()
      ..color = const Color(0xFF7F7CFF).withOpacity(0.6 * (1 - pulseValue)) // 불투명도 상향
      ..style = PaintingStyle.fill;

    canvas.drawCircle(end, 15 + (20 * pulseValue), pulsePaint);
  }

  void _drawDashedLine(Canvas canvas, Offset p1, Offset p2, Paint paint) {
    const dashWidth = 5.0;
    const dashSpace = 5.0;

    double distance = (p2 - p1).distance;
    double currentDistance = 0;
    while (currentDistance < distance) {
      final start = Offset.lerp(p1, p2, currentDistance / distance)!;
      currentDistance += dashWidth;
      final end = Offset.lerp(
        p1,
        p2,
        (currentDistance < distance ? currentDistance : distance) / distance,
      )!;
      canvas.drawLine(start, end, paint);
      currentDistance += dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant _DynamicLinePainter oldDelegate) =>
      oldDelegate.pulseValue != pulseValue ||
      oldDelegate.robotX != robotX ||
      oldDelegate.targetX != targetX;
}

class _AuthMarker extends StatelessWidget {
  final AuthEvent event;
  final MapTransformer transformer;

  const _AuthMarker({required this.event, required this.transformer});

  @override
  Widget build(BuildContext context) {
    if (event.x == null || event.y == null) return const SizedBox.shrink();

    final transformed = transformer.transform(
      event.x!,
      event.y!,
      event.yaw ?? 0,
    );
    final px = transformed['px']!;
    final py = transformed['py']!;

    Color markerColor;
    String statusLabel;
    switch (event.status) {
      case 'waiting_rfid':
        markerColor = const Color(0xFFFACC15);
        statusLabel = '인증 진행 중';
        break;
      case 'success':
        markerColor = const Color(0xFF4ADE80);
        statusLabel = '인증 성공';
        break;
      case 'fail':
        markerColor = const Color(0xFFEF4444);
        statusLabel = '인증 실패';
        break;
      case 'timeout':
        markerColor = const Color(0xFF9FA4B9);
        statusLabel = '미인증';
        break;
      default:
        markerColor = const Color(0xFF9FA4B9);
        statusLabel = '상태 알 수 없음';
        break;
    }

    return Positioned(
      left: px - 12,
      top: py - 12,
      child: Tooltip(
        message: '[$statusLabel] ${event.employeeName ?? ""}',
        child: Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: markerColor.withOpacity(0.15),
            shape: BoxShape.circle,
            border: Border.all(color: markerColor, width: 2),
            boxShadow: [
              BoxShadow(
                color: markerColor.withOpacity(0.3),
                blurRadius: 8,
                spreadRadius: 1,
              ),
            ],
          ),
          child: Icon(
            event.status == 'success' ? Icons.person : Icons.person_search,
            color: markerColor,
            size: 14,
          ),
        ),
      ),
    );
  }
}
