import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../models/event_model.dart';

// 맵 상에서 클릭된 이벤트를 추적하기 위한 로컬 상태
final _selectedMapEventProvider = StateProvider<String?>((ref) => null);

class DataCenterMap extends ConsumerWidget {
  const DataCenterMap({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final alerts = ref.watch(eventListProvider);

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
                  //---------- 이벤트 발생 시 생성되는 마커들 ----------
                  ...alerts.map((event) => _EventMarker(event: event)),
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

  Alignment _getAlignmentForPlace(String placeId) {
    // 장소 ID에 기반한 간단한 해시로 맵 상의 고정 좌표(Alignment) 생성
    final hash = placeId.hashCode;
    final x = (((hash % 100) / 50.0) - 1.0) * 0.8; // -0.8 ~ 0.8 구역 내 배치
    final y = ((((hash ~/ 100) % 100) / 50.0) - 1.0) * 0.8;
    return Alignment(x, y);
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAnomaly = event.anomalyFlag == 1;
    final selectedId = ref.watch(_selectedMapEventProvider);
    final isSelected = selectedId == event.eventId;

    return Align(
      alignment: _getAlignmentForPlace(event.placeId),
      child: Stack(
        clipBehavior: Clip.none,
        alignment: Alignment.center,
        children: [
          // 말풍선 (클릭 시 표시됨)
          if (isSelected)
            Positioned(
              bottom: 24, // 마커 바로 위로 띄움
              child: Container(
                width: 140,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: const Color(0xFF1C1E2B),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: isAnomaly ? const Color(0xFFEF4444) : const Color(0xFF38BDF8),
                  ),
                  boxShadow: const [
                    BoxShadow(color: Colors.black45, blurRadius: 8, offset: Offset(0, 4)),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      event.summaryText ?? '이벤트 발생',
                      style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    // 이미지 영역 (없으면 아이콘)
                    Container(
                      height: 60,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: const Color(0xFF11121A),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: event.frames.isNotEmpty
                          ? Image.network(
                              'http://localhost:8000/images/${event.frames.first.imagePath}',
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => const Icon(Icons.broken_image, color: Colors.grey, size: 24),
                            )
                          : const Icon(Icons.image_not_supported, color: Colors.grey, size: 24),
                    ),
                  ],
                ),
              ),
            ),
          
          // 실제 표시되는 마커 본체
          GestureDetector(
            onTap: () {
              // 선택 토글
              final notifier = ref.read(_selectedMapEventProvider.notifier);
              if (notifier.state == event.eventId) {
                notifier.state = null; // 닫기
              } else {
                notifier.state = event.eventId; // 열기
              }
            },
            child: Container(
              width: isAnomaly ? 20 : 16,
              height: isAnomaly ? 20 : 16,
              decoration: BoxDecoration(
                color: isAnomaly ? const Color(0xFFFF4B5C) : const Color(0xFF38BDF8),
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white, width: 1.5),
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


