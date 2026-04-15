import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/audio_event_model.dart';
import 'server_config_provider.dart';

class AudioEventListNotifier extends StateNotifier<List<AudioEvent>> {
  final Ref ref;
  AudioEventListNotifier(this.ref) : super([]) {
    _startPolling();
  }

  Timer? _timer;

  void _startPolling() {
    _fetchEvents();
    _timer = Timer.periodic(const Duration(seconds: 3), (_) {
      _fetchEvents();
    });
  }

  Future<void> _fetchEvents() async {
    final config = ref.read(serverConfigProvider);
    try {
      final response = await http.get(Uri.parse('${config.baseUrl}/audio_events'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['ok'] == true) {
          final List<dynamic> eventsJson = data['audio_events'] ?? [];
          state = eventsJson.map((e) => AudioEvent.fromJson(e)).toList();
        }
      }
    } catch (e) {
      // polling error ignored
    }
  }

  // 시뮬레이터용 인젝션
  Future<void> injectMockEvent() async {
    final config = ref.read(serverConfigProvider);
    try {
      await http.post(Uri.parse('${config.baseUrl}/test/create_audio_event'));
      await _fetchEvents();
    } catch (e) {
      // ignore
    }
  }

  Future<void> updateLabel(String id, String label) async {
    final config = ref.read(serverConfigProvider);
    // UI 즉각 반영을 위해 로컬 상태 먼저 업데이트
    state = state.map((e) {
      if (e.audioEventId == id) {
        return e.copyWith(adminChecked: 1, adminLabel: label);
      }
      return e;
    }).toList();

    try {
      final response = await http.patch(
        Uri.parse('${config.baseUrl}/audio_events/$id/label'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'admin_label': label}),
      );
      if (response.statusCode == 200) {
        _fetchEvents(); // 백그라운드 동기화
      }
    } catch (e) {
      // 무시
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}


final audioEventListProvider = StateNotifierProvider<AudioEventListNotifier, List<AudioEvent>>((ref) {
  return AudioEventListNotifier(ref);
});
