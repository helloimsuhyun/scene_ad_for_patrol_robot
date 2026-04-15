import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/event_provider.dart';
import '../providers/audio_provider.dart';
import '../providers/yolo_provider.dart';
import '../models/event_model.dart';
import '../models/audio_event_model.dart';
import '../models/yolo_event_model.dart';
import '../providers/server_config_provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

// 포멧용 유틸

String formatEventTime(String isoString) {
  try {
    final dt = DateTime.parse(isoString).toLocal();
    final date = '${dt.year.toString().substring(2)}/${dt.month.toString().padLeft(2, '0')}/${dt.day.toString().padLeft(2, '0')}';
    final time = '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}';
    return '$date $time';
  } catch (_) {
    return '--:--:--';
  }
}

void showEventDetailDialog(BuildContext context, WidgetRef ref, Event event) {
  final config = ref.read(serverConfigProvider);
  ref.read(selectedEventProvider.notifier).state = event;

  final isAlarm = event.anomalyFlag == 1;
  final mainColor = isAlarm ? const Color(0xFFEF4444) : const Color(0xFF38BDF8);

  showDialog(
    context: context,
    builder: (context) {
      return Dialog(
        backgroundColor: const Color(0xFF0D0E15),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: BorderSide(color: mainColor, width: 2),
        ),
        child: Container(
          width: 500,
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    color: mainColor,
                    child: Text(
                      isAlarm ? 'WARNING : 비전 이상 감지' : 'INFO : 시스템 알림',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w900,
                        letterSpacing: 1.0,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                decoration: BoxDecoration(
                  border: Border.all(color: const Color(0xFF2D3041)),
                ),
                child: Image.network(
                  event.frames.isNotEmpty
                      ? '${config.baseUrl}/images/${event.frames.first.imagePath.replaceFirst("recv/", "")}'
                      : 'https://via.placeholder.com/500x300.png?text=No+Image',
                  width: double.infinity,
                  height: 300,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      width: double.infinity,
                      height: 300,
                      color: const Color(0xFF1C1E2B),
                      child: const Center(
                        child: Text(
                          '이미지 연결 실패',
                          style: TextStyle(
                            color: Color(0xFFEF4444),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),
              Text(
                event.summaryText ?? "알 수 없는 이벤트",
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (event.adminLabel != null && event.adminLabel!.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  '상태 : (${event.adminLabel})',
                  style: const TextStyle(
                    color: Color(0xFF4ADE80),
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
              const SizedBox(height: 20),
              const Divider(color: Color(0xFF2D3041), height: 1),
              const SizedBox(height: 16),
              _DetailRow(
                label: '발생 시간',
                value: formatEventTime(event.capturedAt),
              ),
              const SizedBox(height: 12),
              _DetailRow(label: '감시 구역 ID', value: event.placeId),
              if (event.anomalyScore != null) ...[
                const SizedBox(height: 12),
                _DetailRow(
                  label: '위험성 (Confidence)',
                  value: '${(event.anomalyScore! * 100).toStringAsFixed(1)}%',
                  valueColor: mainColor,
                ),
              ],
              const SizedBox(height: 16),
              const Divider(color: Color(0xFF2D3041), height: 1),
              const SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(
                          0xFF22C55E,
                        ).withOpacity(0.15),
                        foregroundColor: const Color(0xFF22C55E),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(4),
                        ),
                        elevation: 0,
                      ),
                      child: const Text(
                        '정상 (오탐지 - 뱅크 추가)',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      onPressed: () async {
                        try {
                          Navigator.of(context).pop();
                          final response = await http.post(
                            Uri.parse('${config.baseUrl}/move_event'),
                            headers: {'Content-Type': 'application/json'},
                            body:
                                '{"event_id": "${event.eventId}", "move": true}',
                          );
                          if (response.statusCode == 200) {
                            ref.invalidate(eventListProvider);
                          }
                        } catch (e) {
                          debugPrint('Error: $e');
                        }
                      },
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFFEF4444),
                        side: const BorderSide(color: Color(0xFFEF4444)),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                      child: const Text(
                        '파손/침입 경보',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      onPressed: () async {
                        Navigator.of(context).pop();
                        try {
                          await http.patch(
                            Uri.parse(
                              '${config.baseUrl}/events/${event.eventId}/label',
                            ),
                            headers: {'Content-Type': 'application/json'},
                            body: '{"admin_label": "파손/침입"}',
                          );
                          ref.invalidate(eventListProvider);
                        } catch (e) {
                          debugPrint('Label error: $e');
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton(
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFFF97316),
                        side: const BorderSide(color: Color(0xFFF97316)),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ),
                      child: const Text(
                        '기타 이상 현상',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      onPressed: () async {
                        Navigator.of(context).pop();
                        try {
                          await http.patch(
                            Uri.parse(
                              '${config.baseUrl}/events/${event.eventId}/label',
                            ),
                            headers: {'Content-Type': 'application/json'},
                            body: '{"admin_label": "기타 이상"}',
                          );
                          ref.invalidate(eventListProvider);
                        } catch (e) {
                          debugPrint('Label error: $e');
                        }
                      },
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      );
    },
  );
}

//---------- 오디오 이벤트 상세 팝업 ----------
void showAudioEventDetailDialog(
  BuildContext context,
  WidgetRef ref,
  AudioEvent event,
) {
  final mainColor = const Color(0xFFBA68C8);
  showDialog(
    context: context,
    builder: (context) {
      return Dialog(
        backgroundColor: const Color(0xFF0D0E15),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: BorderSide(color: mainColor, width: 2),
        ),
        child: Container(
          width: 450,
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    color: mainColor,
                    child: const Text(
                      'WARNING : 오디오 이상 감지',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  border: Border.all(color: const Color(0xFF2D3041)),
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning, color: mainColor, size: 32),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            event.modelLabel ?? '미분류 소리',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '발생 각도: ${event.doa?.toStringAsFixed(1) ?? "--"}°',
                            style: const TextStyle(
                              color: Colors.white60,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              if (event.adminLabel != null && event.adminLabel!.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  '상태 : (${event.adminLabel})',
                  style: const TextStyle(
                    color: Color(0xFF4ADE80),
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
              const SizedBox(height: 12),
              _DetailRow(
                label: '발생 시간',
                value: formatEventTime(event.timestamp),
              ),
              if (event.adminLabel == null || event.adminLabel!.isEmpty) ...[
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      ref
                          .read(audioEventListProvider.notifier)
                          .updateLabel(event.audioEventId ?? '', '확인완료');
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: mainColor,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    child: const Text(
                      '경보 해제',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      );
    },
  );
}

//---------- YOLO 사람 감지 상세 팝업 ----------
void showYoloEventDetailDialog(
  BuildContext context,
  WidgetRef ref,
  YoloEvent event,
) {
  final config = ref.read(serverConfigProvider);
  final mainColor = const Color(0xFFEAB308);

  showDialog(
    context: context,
    builder: (context) {
      return Dialog(
        backgroundColor: const Color(0xFF0D0E15),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
          side: BorderSide(color: mainColor, width: 2),
        ),
        child: Container(
          width: 500,
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 4,
                    ),
                    color: mainColor,
                    child: const Text(
                      'WARNING : 보안 구역 인물 감지',
                      style: TextStyle(
                        color: Colors.black,
                        fontSize: 15,
                        fontWeight: FontWeight.w900,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              if (event.imageUrl != null || event.imagePath != null)
                Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: const Color(0xFF2D3041)),
                  ),
                  child: Image.network(
                    event.imageUrl ??
                        '${config.baseUrl}/person_images/${event.imagePath!.replaceFirst("recv_person/", "")}',
                    width: double.infinity,
                    height: 300,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => Container(
                      height: 200,
                      color: const Color(0xFF1C1E2B),
                      child: const Center(
                        child: Text(
                          '이미지 데이터 없음',
                          style: TextStyle(
                            color: Color(0xFFEF4444),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              const SizedBox(height: 16),
              Text(
                event.eventType ?? "인물 감지됨",
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (event.adminLabel != null && event.adminLabel!.isNotEmpty) ...[
                const SizedBox(height: 8),
                Text(
                  '상태 : (${event.adminLabel})',
                  style: const TextStyle(
                    color: Color(0xFF4ADE80),
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
              const SizedBox(height: 12),
              _DetailRow(
                label: '발생 시간',
                value: formatEventTime(event.timestamp),
              ),
              _DetailRow(label: '감지 인원', value: '${event.personCount} 명'),
              if (event.sourceRegionName != null)
                _DetailRow(label: '감시 구역', value: event.sourceRegionName!),
              if (event.dwellTimeSec != null)
                _DetailRow(
                  label: '체류 시간',
                  value: '${event.dwellTimeSec!.toStringAsFixed(1)} 초',
                ),

              if (event.adminLabel == null || event.adminLabel!.isEmpty) ...[
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.pop(context);
                      ref
                          .read(yoloEventsProvider.notifier)
                          .updateLabel(event.yoloEventId ?? '', '확인완료');
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: mainColor,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                    child: const Text(
                      '경보 해제',
                      style: TextStyle(
                        color: Colors.black,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      );
    },
  );
}

class _DetailRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;

  const _DetailRow({required this.label, required this.value, this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 13),
          ),
          Text(
            value,
            style: TextStyle(
              color: valueColor ?? Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
