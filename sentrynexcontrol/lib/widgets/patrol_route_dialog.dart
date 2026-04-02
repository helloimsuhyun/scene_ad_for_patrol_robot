import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../providers/map_provider.dart';
import '../features/control/control_provider.dart';

class PatrolRouteDialog extends ConsumerStatefulWidget {
  const PatrolRouteDialog({super.key});

  @override
  ConsumerState<PatrolRouteDialog> createState() => _PatrolRouteDialogState();
}

class _PatrolRouteDialogState extends ConsumerState<PatrolRouteDialog> {
  // 사용자가 선택한 장소의 순서를 기록
  List<String> selectedPlaceIds = [];

  bool isSubmitting = false;
  bool _isEditMode = false;

  @override
  Widget build(BuildContext context) {
    final placesAsync = ref.watch(placesProvider);
    final mapImagePathAsync = ref.watch(mapImagePathProvider);
    final mapTransformerAsync = ref.watch(mapTransformerProvider);

    return Dialog(
      backgroundColor: const Color(0xFF161822),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        width: 1100,
        height: 700,
        clipBehavior: Clip.antiAlias,
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(16)),
        child: Row(
          children: [
            //---------- 좌측: 대화형 맵 에리어 ----------
            Expanded(
              flex: 7,
              child: Container(
                color: const Color(0xFF10121A),
                child: mapImagePathAsync.when(
                  data: (imagePath) => mapTransformerAsync.when(
                    data: (transformer) {
                      return placesAsync.when(
                        data: (placesData) {
                          final places =
                              placesData['places'] as List<dynamic>? ?? [];

                          // 선택된 노드들의 픽셀 좌표 리스트 추출 (점선 연결용)
                          final List<Offset> points = [];
                          final Map<String, dynamic> placeMap = {
                            for (var p in places) p['place_id'].toString(): p,
                          };

                          for (final id in selectedPlaceIds) {
                            final p = placeMap[id];
                            if (p != null) {
                              double px = 0, py = 0;
                              if (p['x'] != null && p['y'] != null) {
                                final pt = transformer.transform(
                                  p['x'],
                                  p['y'],
                                  p['yaw'] ?? 0,
                                );
                                px = pt['px']!;
                                py = pt['py']!;
                              } else {
                                final hash = id.hashCode;
                                px = (hash % 1000) + 100.0;
                                py = ((hash ~/ 1000) % 1000) + 100.0;
                              }
                              points.add(Offset(px, py));
                            }
                          }

                          return InteractiveViewer(
                            minScale: 1.0,
                            maxScale: 5.0,
                            child: Center(
                              child: FittedBox(
                                fit: BoxFit.contain,
                                alignment: Alignment.center,
                                child: GestureDetector(
                                  onTapUp: (details) async {
                                    if (_isEditMode) return;
                                    final px = details.localPosition.dx;
                                    final py = details.localPosition.dy;
                                    final pt = transformer.inverseTransform(
                                      px,
                                      py,
                                      0,
                                    );

                                    final nameController =
                                        TextEditingController();
                                    final confirmed = await showDialog<bool>(
                                      context: context,
                                      builder: (ctx) => AlertDialog(
                                        backgroundColor: const Color(
                                          0xFF1C1E2B,
                                        ),
                                        title: const Text(
                                          '새 위치 노드 생성',
                                          style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 16,
                                          ),
                                        ),
                                        content: TextField(
                                          controller: nameController,
                                          style: const TextStyle(
                                            color: Colors.white,
                                          ),
                                          decoration: const InputDecoration(
                                            hintText: '장소 이름 (예: 복도1)',
                                            hintStyle: TextStyle(
                                              color: Colors.white54,
                                            ),
                                          ),
                                        ),
                                        actions: [
                                          TextButton(
                                            onPressed: () =>
                                                Navigator.pop(ctx, false),
                                            child: const Text('취소'),
                                          ),
                                          ElevatedButton(
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: const Color(
                                                0xFF7F7CFF,
                                              ),
                                            ),
                                            onPressed: () {
                                              if (nameController
                                                  .text
                                                  .isNotEmpty)
                                                Navigator.pop(ctx, true);
                                            },
                                            child: const Text(
                                              '생성',
                                              style: TextStyle(
                                                color: Colors.white,
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    );

                                    if (confirmed == true && context.mounted) {
                                      final newId =
                                          "P_${DateTime.now().millisecondsSinceEpoch}";
                                      final res = await http.post(
                                        Uri.parse(
                                          'http://127.0.0.1:8000/places',
                                        ),
                                        headers: {
                                          'Content-Type': 'application/json',
                                        },
                                        body: jsonEncode({
                                          "place_id": newId,
                                          "display_name": nameController.text,
                                          "x": pt['x'],
                                          "y": pt['y'],
                                          "yaw": 0.0,
                                          "patrol_enabled": false,
                                        }),
                                      );
                                      if (res.statusCode == 200) {
                                        ref.invalidate(placesProvider);
                                        ScaffoldMessenger.of(
                                          context,
                                        ).showSnackBar(
                                          const SnackBar(
                                            content: Text('새 노드가 화면에 생성되었습니다!'),
                                          ),
                                        );
                                      }
                                    }
                                  },
                                  child: Stack(
                                    clipBehavior: Clip.none,
                                    children: [
                                      Padding(
                                        padding: const EdgeInsets.only(
                                          right: 60,
                                        ),
                                        child: Image.asset(imagePath),
                                      ),
                                      Positioned.fill(
                                        child: IgnorePointer(
                                          child: CustomPaint(
                                            painter: _DottedLinePainter(
                                              points: points,
                                            ),
                                          ),
                                        ),
                                      ),
                                      ...places.map((p) {
                                        final placeId = p['place_id']
                                            .toString();
                                        final displayName =
                                            p['display_name']?.toString() ??
                                            placeId;

                                        double px = 0, py = 0;
                                        if (p['x'] != null && p['y'] != null) {
                                          final pt = transformer.transform(
                                            p['x'],
                                            p['y'],
                                            p['yaw'] ?? 0,
                                          );
                                          px = pt['px']!;
                                          py = pt['py']!;
                                        } else {
                                          final hash = placeId.hashCode;
                                          px = (hash % 1000) + 100.0;
                                          py = ((hash ~/ 1000) % 1000) + 100.0;
                                        }

                                        final isSelected = selectedPlaceIds
                                            .contains(placeId);
                                        final orderIndex =
                                            selectedPlaceIds.indexOf(placeId) +
                                            1;

                                        return Positioned(
                                          left: px - 20,
                                          top: py - 20,
                                          child: GestureDetector(
                                            behavior: HitTestBehavior.opaque,
                                            onTap: () async {
                                              if (_isEditMode) {
                                                final renameController =
                                                    TextEditingController(
                                                      text: displayName,
                                                    );
                                                final confirmed = await showDialog<bool>(
                                                  context: context,
                                                  builder: (ctx) => AlertDialog(
                                                    backgroundColor:
                                                        const Color(0xFF1C1E2B),
                                                    title: const Text(
                                                      '노드 이름 변경',
                                                      style: TextStyle(
                                                        color: Colors.white,
                                                        fontSize: 16,
                                                      ),
                                                    ),
                                                    content: TextField(
                                                      controller:
                                                          renameController,
                                                      style: const TextStyle(
                                                        color: Colors.white,
                                                      ),
                                                      decoration:
                                                          const InputDecoration(
                                                            hintStyle:
                                                                TextStyle(
                                                                  color: Colors
                                                                      .white54,
                                                                ),
                                                          ),
                                                    ),
                                                    actions: [
                                                      TextButton(
                                                        onPressed: () =>
                                                            Navigator.pop(
                                                              ctx,
                                                              false,
                                                            ),
                                                        child: const Text('취소'),
                                                      ),
                                                      ElevatedButton(
                                                        style:
                                                            ElevatedButton.styleFrom(
                                                              backgroundColor:
                                                                  const Color(
                                                                    0xFF7F7CFF,
                                                                  ),
                                                            ),
                                                        onPressed: () {
                                                          if (renameController
                                                              .text
                                                              .isNotEmpty)
                                                            Navigator.pop(
                                                              ctx,
                                                              true,
                                                            );
                                                        },
                                                        child: const Text(
                                                          '저장',
                                                          style: TextStyle(
                                                            color: Colors.white,
                                                          ),
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                );
                                                if (confirmed == true &&
                                                    context.mounted) {
                                                  await ControlActions.updateDisplayName(
                                                    ref,
                                                    placeId,
                                                    renameController.text,
                                                  );
                                                }
                                              } else {
                                                setState(() {
                                                  if (isSelected) {
                                                    selectedPlaceIds.remove(
                                                      placeId,
                                                    );
                                                  } else {
                                                    selectedPlaceIds.add(
                                                      placeId,
                                                    );
                                                  }
                                                });
                                              }
                                            },
                                            child: Stack(
                                              clipBehavior: Clip.none,
                                              children: [
                                                Column(
                                                  mainAxisSize:
                                                      MainAxisSize.min,
                                                  children: [
                                                    Container(
                                                      width: 32,
                                                      height: 32,
                                                      decoration: BoxDecoration(
                                                        color:
                                                            Colors.transparent,
                                                        shape: BoxShape.circle,
                                                        border: Border.all(
                                                          color: const Color(
                                                            0xFF7F7CFF,
                                                          ),
                                                          width: 2,
                                                        ),
                                                        boxShadow: isSelected
                                                            ? const [
                                                                BoxShadow(
                                                                  color: Color(
                                                                    0x667F7CFF,
                                                                  ),
                                                                  blurRadius:
                                                                      10,
                                                                  spreadRadius:
                                                                      2,
                                                                ),
                                                              ]
                                                            : [],
                                                      ),
                                                      child: Center(
                                                        child: isSelected
                                                            ? Container(
                                                                width: 25,
                                                                height: 25,
                                                                decoration: BoxDecoration(
                                                                  color: const Color(
                                                                    0xFF7F7CFF,
                                                                  ),
                                                                  shape: BoxShape
                                                                      .circle,
                                                                ),
                                                                child: Center(
                                                                  child: Text(
                                                                    '$orderIndex',
                                                                    style: const TextStyle(
                                                                      color: Colors
                                                                          .white,
                                                                      fontWeight:
                                                                          FontWeight
                                                                              .bold,
                                                                      fontSize:
                                                                          11,
                                                                    ),
                                                                  ),
                                                                ),
                                                              )
                                                            : const SizedBox.shrink(),
                                                      ),
                                                    ),
                                                    const SizedBox(height: 4),
                                                    Container(
                                                      // 노드 이름표
                                                      padding:
                                                          const EdgeInsets.symmetric(
                                                            horizontal: 6,
                                                            vertical: 4, // 2 -> 4로 확대
                                                          ),
                                                      decoration: BoxDecoration(
                                                        color: Colors.black87,
                                                        borderRadius:
                                                            BorderRadius.circular(
                                                              4,
                                                            ),
                                                      ),
                                                      child: Text(
                                                        displayName,
                                                        style: const TextStyle(
                                                          color: Colors.white,
                                                          fontSize: 10,
                                                          fontWeight:
                                                              FontWeight.w600,
                                                        ),
                                                      ),
                                                    ),
                                                  ],
                                                ),
                                                if (_isEditMode)
                                                  Positioned(
                                                    top: -8,
                                                    right: -8,
                                                    child: GestureDetector(
                                                      onTap: () async {
                                                        final confirmed = await showDialog<bool>(
                                                          context: context,
                                                          builder: (ctx) => AlertDialog(
                                                            backgroundColor:
                                                                const Color(
                                                                  0xFF1C1E2B,
                                                                ),
                                                            title: const Text(
                                                              '노드 영구 삭제',
                                                              style: TextStyle(
                                                                color: Colors
                                                                    .white,
                                                                fontSize: 16,
                                                              ),
                                                            ),
                                                            content: Text(
                                                              "'$displayName' 노드를 DB에서 영구적으로 삭제하겠습니까?",
                                                              style:
                                                                  const TextStyle(
                                                                    color: Colors
                                                                        .white70,
                                                                  ),
                                                            ),
                                                            actions: [
                                                              TextButton(
                                                                onPressed: () =>
                                                                    Navigator.pop(
                                                                      ctx,
                                                                      false,
                                                                    ),
                                                                child:
                                                                    const Text(
                                                                      '취소',
                                                                    ),
                                                              ),
                                                              ElevatedButton(
                                                                style: ElevatedButton.styleFrom(
                                                                  backgroundColor:
                                                                      Colors
                                                                          .redAccent,
                                                                ),
                                                                onPressed: () =>
                                                                    Navigator.pop(
                                                                      ctx,
                                                                      true,
                                                                    ),
                                                                child: const Text(
                                                                  '삭제',
                                                                  style: TextStyle(
                                                                    color: Colors
                                                                        .white,
                                                                  ),
                                                                ),
                                                              ),
                                                            ],
                                                          ),
                                                        );

                                                        if (confirmed == true &&
                                                            context.mounted) {
                                                          await ControlActions.deletePlace(
                                                            ref,
                                                            placeId,
                                                          );
                                                          ref.invalidate(
                                                            placesProvider,
                                                          );
                                                          setState(() {
                                                            selectedPlaceIds
                                                                .remove(
                                                                  placeId,
                                                                );
                                                          });
                                                        }
                                                      },
                                                      child: Container(
                                                        padding:
                                                            const EdgeInsets.all(
                                                              3,
                                                            ),
                                                        decoration:
                                                            BoxDecoration(
                                                              color: Colors
                                                                  .redAccent
                                                                  .withAlpha(
                                                                    70,
                                                                  ),
                                                              shape: BoxShape
                                                                  .circle,
                                                            ),
                                                        child: const Icon(
                                                          Icons.close,
                                                          size: 14,
                                                          color:
                                                              Colors.redAccent,
                                                        ),
                                                      ),
                                                    ),
                                                  ),
                                              ],
                                            ),
                                          ),
                                        );
                                      }),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          );
                        },
                        loading: () =>
                            const Center(child: CircularProgressIndicator()),
                        error: (err, stack) => Center(
                          child: Text(
                            'Places Error: $err',
                            style: const TextStyle(color: Colors.red),
                          ),
                        ),
                      );
                    },
                    loading: () =>
                        const Center(child: CircularProgressIndicator()),
                    error: (err, stack) => Center(
                      child: Text(
                        'Map Transform Error: $err',
                        style: const TextStyle(color: Colors.red),
                      ),
                    ),
                  ),
                  loading: () =>
                      const Center(child: CircularProgressIndicator()),
                  error: (err, stack) => Center(
                    child: Text(
                      'Map Image Error: $err',
                      style: const TextStyle(color: Colors.red),
                    ),
                  ),
                ),
              ),
            ),
            //---------- 우측: 사이드 패널 ----------
            Expanded(
              flex: 3,
              child: Container(
                color: const Color(0xFF161822),
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          '순찰 루트 설정',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Row(
                          children: [
                            TextButton.icon(
                              icon: Icon(
                                _isEditMode ? Icons.check : Icons.edit,
                                color: _isEditMode
                                    ? Colors.green
                                    : Colors.white54,
                                size: 18,
                              ),
                              label: Text(
                                _isEditMode ? '편집 완료' : '노드 편집',
                                style: TextStyle(
                                  color: _isEditMode
                                      ? Colors.green
                                      : Colors.white54,
                                ),
                              ),
                              onPressed: () {
                                setState(() {
                                  _isEditMode = !_isEditMode;
                                });
                              },
                            ),
                            IconButton(
                              icon: const Icon(
                                Icons.close,
                                color: Colors.white54,
                              ),
                              onPressed: () => Navigator.of(context).pop(),
                            ),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _isEditMode
                          ? '노드를 탭하여 이름을 변경하거나 X 아이콘을 눌러 영구 삭제하세요.'
                          : '맵 노드를 클릭하여 경로에 추가하세요.\n(원하는 위치를 클릭하여 새 노드 생성)',
                      style: const TextStyle(
                        color: Color(0xFF9FA4B9),
                        fontSize: 12,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // 선택된 리스트
                    Expanded(
                      child: placesAsync.when(
                        data: (placesData) {
                          final places =
                              placesData['places'] as List<dynamic>? ?? [];
                          final Map<String, dynamic> placeMap = {
                            for (var p in places) p['place_id'].toString(): p,
                          };

                          if (selectedPlaceIds.isEmpty) {
                            return const Center(
                              child: Text(
                                '선택된 노드가 없습니다.',
                                style: TextStyle(color: Colors.white24),
                              ),
                            );
                          }

                          return ListView.builder(
                            itemCount: selectedPlaceIds.length,
                            itemBuilder: (context, index) {
                              final id = selectedPlaceIds[index];
                              final displayName =
                                  placeMap[id]?['display_name']?.toString() ??
                                  id;

                              return Container(
                                margin: const EdgeInsets.only(bottom: 8),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 14,
                                ),
                                decoration: BoxDecoration(
                                  color: const Color(0xFF10121A),
                                  borderRadius: BorderRadius.circular(8),
                                  border: Border.all(
                                    color: const Color(0xFF2D3041),
                                  ),
                                ),
                                child: Row(
                                  children: [
                                    Container(
                                      width: 24,
                                      height: 24,
                                      decoration: BoxDecoration(
                                        color: const Color(0xFF26293A),
                                        borderRadius: BorderRadius.circular(4),
                                      ),
                                      child: Center(
                                        child: Text(
                                          '${index + 1}',
                                          style: const TextStyle(
                                            color: Color(0xFFB5BAD3),
                                            fontSize: 11,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: Text(
                                        displayName,
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 14,
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ),
                                    InkWell(
                                      onTap: () {
                                        setState(() {
                                          selectedPlaceIds.removeAt(index);
                                        });
                                      },
                                      child: const Icon(
                                        Icons.remove_circle_outline,
                                        color: Color(0xFF9FA4B9),
                                        size: 20,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          );
                        },
                        loading: () => const SizedBox.shrink(),
                        error: (_, __) => const SizedBox.shrink(),
                      ),
                    ),

                    const SizedBox(height: 16),
                    // 하단 예상 시간 및 시작 버튼
                    Text(
                      '예상 소요시간 : ${selectedPlaceIds.length * 5}분', // 임의 계산
                      style: const TextStyle(
                        color: Color(0xFF9FA4B9),
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: selectedPlaceIds.isNotEmpty
                              ? const Color(0xFF7F7CFF)
                              : const Color(0xFF2D3041),
                          padding: const EdgeInsets.symmetric(vertical: 18),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                        onPressed: (selectedPlaceIds.isEmpty || isSubmitting)
                            ? null
                            : () async {
                                setState(() {
                                  isSubmitting = true;
                                });
                                final places =
                                    placesAsync.value?['places']
                                        as List<dynamic>? ??
                                    [];
                                final List<Map<String, dynamic>> allPlaces =
                                    places
                                        .map(
                                          (e) => Map<String, dynamic>.from(e),
                                        )
                                        .toList();

                                await ControlActions.applyRouteAndStart(
                                  ref,
                                  allPlaces,
                                  selectedPlaceIds,
                                );

                                if (context.mounted) {
                                  Navigator.of(context).pop();
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('새로운 순찰 경로로 순찰을 시작합니다.'),
                                    ),
                                  );
                                }
                              },
                        icon: isSubmitting
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 2,
                                ),
                              )
                            : const Icon(Icons.play_arrow, color: Colors.white),
                        label: Text(
                          isSubmitting ? '명령 전송 중...' : '순찰 시작',
                          style: const TextStyle(
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
            ),
          ],
        ),
      ),
    );
  }
}

class _DottedLinePainter extends CustomPainter {
  final List<Offset> points;
  _DottedLinePainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 2) return;
    final paint = Paint()
      ..color = const Color(0xFF7F7CFF)
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    for (int i = 0; i < points.length - 1; i++) {
      _drawDottedLine(canvas, points[i], points[i + 1], paint);
    }
  }

  void _drawDottedLine(Canvas canvas, Offset p1, Offset p2, Paint paint) {
    const double dashWidth = 5;
    const double dashSpace = 5;
    double startX = p1.dx;
    double startY = p1.dy;
    final double distance = (p2 - p1).distance;
    double currentDistance = 0;

    final double dx = (p2.dx - p1.dx) / distance;
    final double dy = (p2.dy - p1.dy) / distance;

    while (currentDistance < distance) {
      final double endX = startX + dx * dashWidth;
      final double endY = startY + dy * dashWidth;

      if (currentDistance + dashWidth > distance) {
        canvas.drawLine(Offset(startX, startY), p2, paint);
      } else {
        canvas.drawLine(Offset(startX, startY), Offset(endX, endY), paint);
      }
      startX += dx * (dashWidth + dashSpace);
      startY += dy * (dashWidth + dashSpace);
      currentDistance += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(_DottedLinePainter oldDelegate) => true;
}
