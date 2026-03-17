import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';
import 'event_detail_dialog.dart';

// 맵 상에서 클릭된 이벤트를 추적하기 위한 로컬 상태
final _selectedMapEventProvider = StateProvider<String?>((ref) => null);

class DataCenterMap extends ConsumerWidget {
  const DataCenterMap({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final alerts = ref.watch(eventListProvider);

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
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  //---------- 맵 배경 격자(Grid) 무늬 레이어 ----------
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _MapGridPainter(),
                    ),
                  ),
                  //---------- 각 장소별 최신 이벤트 마커 표시 ----------
                  ...latestEventsByPlace.values.map((event) => _EventMarker(event: event)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _EventMarker extends ConsumerWidget {
  final Event event;
  const _EventMarker({required this.event});

  //---------- 특정 장소 ID별로 고정된 지도 상의 위치 (원하는 위치로 수정 가능) ----------
  static const Map<String, Alignment> _fixedPositions = {
    '00': Alignment(-0.85, -0.65), // 00번 장소
    '01': Alignment(0.75, 0.45),   // 01번 장소
    '02': Alignment(0.0, 0.85),    // 02번 장소
  };

  Alignment _getAlignmentForPlace(String placeId) {
    // 1. 먼저 고정된 위치값이 있는지 확인
    if (_fixedPositions.containsKey(placeId)) {
      return _fixedPositions[placeId]!;
    }

    // 2. 등록되지 않은 ID라면 장소 ID에 기반한 간단한 해시로 맵 상의 고정 좌표 생성
    final hash = placeId.hashCode;
    final x = (((hash % 100) / 50.0) - 1.0) * 0.9;
    final y = ((((hash ~/ 100) % 100) / 50.0) - 1.0) * 0.9;
    return Alignment(x, y);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAnomaly = event.anomalyFlag == 1;
    final selectedId = ref.watch(_selectedMapEventProvider);
    final isSelected = selectedId == event.eventId;

    //---------- 히트 테스트(터치 인식) 영역 문제 해결 ----------
    // 히트 테스트 영역을 확보하면서 마커 위치가 튀지 않도록 전체 크기를 항상 일정하게(220x320) 고정합니다.
    // (Align 위젯은 자식의 가로/세로 크기가 변하면 중심 좌표를 다시 계산하기 때문에 위치가 튀는 버그가 발생함)
    return Align(
      alignment: _getAlignmentForPlace(event.placeId),
      child: SizedBox(
        width: 220, 
        height: 320,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center, // 마커를 박스 정중앙(중심점)에 배치
          children: [
            // 말풍선 (선택 시 마커 상단에 표시)
            if (isSelected)
              Positioned(
                bottom: 160 + 15, // 박스 중앙(160) + 마커 높이 절반 이상 여유
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
                                    'http://172.17.78.222:8000/images/${event.frames.first.imagePath.replaceFirst("recv/", "")}',
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
                      ],
                    ),
                  ),
                ),
              ),
            
            // 실제 표시되는 마커 본체 (중앙 유지)
            GestureDetector(
              onTap: () {
                final notifier = ref.read(_selectedMapEventProvider.notifier);
                notifier.state = (notifier.state == event.eventId) ? null : event.eventId;
              },
              child: Container(
                width: isAnomaly ? 20 : 16,
                height: isAnomaly ? 20 : 16,
                decoration: BoxDecoration(
                  color: isAnomaly ? const Color(0xFFFF4B5C) : const Color(0xFF38BDF8),
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 2),
                  boxShadow: [
                    BoxShadow(
                      color: isAnomaly ? const Color(0x66FF4B5C) : const Color(0x6638BDF8),
                      blurRadius: 10,
                      spreadRadius: 2,
                    )
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

class _MapGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF26293A)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    const double step = 40;

    for (double x = 0; x <= size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y <= size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
