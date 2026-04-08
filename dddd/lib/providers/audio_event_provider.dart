// audio_event_provider.dart
// 서버의 /audio_events 엔드포인트를 폴링하여 미확인 오디오 이벤트 목록을 갱신하는 Provider
// unchecked_only=true : admin_label이 없는 이벤트만 가져옴 (라벨링 완료된 건 제외)

import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/audio_event_model.dart';

// 현재 환경에 맞춰 서버 주소를 설정하세요 (localhost 또는 실제 서버 IP)
const String _baseUrl = 'http://localhost:8000';

class AudioEventListNotifier extends StateNotifier<List<AudioEventData>> {
  AudioEventListNotifier() : super([]);

  Timer? _pollingTimer;
  bool _isDisposed = false;

  /// 폴링 시작: 5초마다 서버에서 미확인 오디오 이벤트를 가져옴
  void startPolling() {
    _fetchAudioEvents();
    _pollingTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      _fetchAudioEvents();
    });
  }

  /// GET /audio_events?unchecked_only=true
  /// admin_label이 없는 이벤트만 반환하도록 요청
  Future<void> _fetchAudioEvents() async {
    if (_isDisposed) return;

    try {
      final uri = Uri.parse('$_baseUrl/audio_events?unchecked_only=true&limit=100');
      print('DEBUG: Fetching audio events from $uri');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['ok'] == true) {
          final List<dynamic> eventsJson = data['audio_events'];
          print('DEBUG: Received ${eventsJson.length} audio events from $_baseUrl');
          state = eventsJson.map((json) => AudioEventData.fromJson(json)).toList();
        } else {
          print('DEBUG: audio_events ok=false: ${response.body}');
        }
      } else {
        print('DEBUG: audio_events failed status: ${response.statusCode}');
      }
    } catch (e) {
      print('DEBUG: audio_events error connection to $_baseUrl: $e');
    }
  }

  /// PATCH /audio_events/{audio_event_id}/label
  /// 관리자 라벨을 서버에 전송, 성공 시 로컬 상태에서도 해당 마커를 즉시 제거
  Future<void> applyAdminLabel({
    required String audioEventId,
    required String adminLabel,
  }) async {
    if (_isDisposed) return;

    try {
      final uri = Uri.parse('$_baseUrl/audio_events/$audioEventId/label');
      final response = await http.patch(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'admin_label': adminLabel}),
      );

      if (response.statusCode == 200) {
        // 서버 업데이트 성공 → 로컬 목록에서 즉시 제거 (맵에서 사라짐)
        state = state.where((e) => e.audioEventId != audioEventId).toList();
      }
    } catch (e) {
      // 오류 무시
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _pollingTimer?.cancel();
    super.dispose();
  }
}

// 전역 Provider
final audioEventListProvider =
    StateNotifierProvider<AudioEventListNotifier, List<AudioEventData>>((ref) {
  final notifier = AudioEventListNotifier();
  notifier.startPolling();
  return notifier;
});
