import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/yolo_event_model.dart';
import 'server_config_provider.dart';

class YoloEventListNotifier extends StateNotifier<List<YoloEvent>> {
  final Ref ref;
  YoloEventListNotifier(this.ref) : super([]);

  Timer? _pollingTimer;
  String? _lastCapturedAt;
  bool _isDisposed = false;

  void startPolling() {
    _fetchEvents();
    _pollingTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      _fetchEvents();
    });
  }

  Future<void> _fetchEvents() async {
    if (_isDisposed) return;

    final config = ref.read(serverConfigProvider);
    try {
      final uri = _lastCapturedAt != null
          ? Uri.parse('${config.baseUrl}/yolo_events?since=$_lastCapturedAt')
          : Uri.parse('${config.baseUrl}/yolo_events');

      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['ok'] == true) {
          final List<dynamic> eventsJson = data['yolo_events'];
          if (eventsJson.isEmpty) return;

          final newEvents =
              eventsJson.map((json) => YoloEvent.fromJson(json)).toList();

          if (newEvents.isNotEmpty) {
            _lastCapturedAt = newEvents.first.timestamp;
          }

          final updatedState = List<YoloEvent>.from(state);
          for (var newEvent in newEvents) {
            final index = updatedState.indexWhere((e) => e.yoloEventId == newEvent.yoloEventId);
            if (index != -1) {
              updatedState[index] = newEvent;
            } else {
              updatedState.insert(0, newEvent);
            }
          }
          state = updatedState;
        }
      }
    } catch (e) {
      // 무시
    }
  }

  // 시뮬레이터용 인젝션
  Future<void> injectMockEvent() async {
    final config = ref.read(serverConfigProvider);
    try {
      await http.post(Uri.parse('${config.baseUrl}/test/create_yolo_event'));
      await _fetchEvents();
    } catch (e) {
      // ignore
    }
  }

  Future<void> updateLabel(String id, String label) async {
    final config = ref.read(serverConfigProvider);
    
    // UI 즉각 반영을 위해 로컬 상태 먼저 업데이트
    state = state.map((e) {
      if (e.yoloEventId == id) {
        return e.copyWith(adminChecked: 1, adminLabel: label);
      }
      return e;
    }).toList();

    try {
      final response = await http.patch(
        Uri.parse('${config.baseUrl}/yolo_events/$id/label'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'admin_label': label}),
      );
      if (response.statusCode == 200) {
        _fetchEvents(); // 서버와 상태 동기화
      }
    } catch (e) {
      // 무시
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _pollingTimer?.cancel();
    super.dispose();
  }
}

final yoloEventsProvider =
    StateNotifierProvider<YoloEventListNotifier, List<YoloEvent>>((ref) {
  final notifier = YoloEventListNotifier(ref);
  notifier.startPolling();
  return notifier;
});
