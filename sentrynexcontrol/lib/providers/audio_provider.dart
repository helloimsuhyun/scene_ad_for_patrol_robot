import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/audio_event_model.dart';

final String audioBaseUrl = 'http://127.0.0.1:8000';

class AudioEventListNotifier extends StateNotifier<List<AudioEvent>> {
  AudioEventListNotifier() : super([]) {
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
    try {
      final response = await http.get(Uri.parse('$audioBaseUrl/audio_events'));
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

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final audioEventListProvider = StateNotifierProvider<AudioEventListNotifier, List<AudioEvent>>((ref) {
  return AudioEventListNotifier();
});
