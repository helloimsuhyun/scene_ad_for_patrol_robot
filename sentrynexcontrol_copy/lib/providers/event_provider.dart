import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/event_model.dart';

//---------- 서버 주소 설정 (인터넷이 없는 로컬 WiFi 환경의 경우) ----------
// 1. 같은 기기에서 브라우저를 띄울 때는 'localhost'가 작동합니다.
// 2. 태블릿 등 외부 기기에서 접속 시 서버 PC의 IP(예: '192.168.x.x')로 수정이 필요합니다.
const String _baseUrl = 'http://localhost:8000';

class EventListNotifier extends StateNotifier<List<Event>> {
  EventListNotifier() : super([]);

  Timer? _pollingTimer;
  String? _lastCapturedAt;
  bool _isDisposed = false;

  /// 서버에서 이벤트를 3초마다 계속 긁어오도록 설정 (Polling)
  void startPolling() {
    _fetchEvents();

    _pollingTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      _fetchEvents();
    });
  }

  /// 서버에 HTTP GET 요청을 보내 새 이벤트를 가져옵니다.
  Future<void> _fetchEvents() async {
    if (_isDisposed) return;

    try {
      final uri = _lastCapturedAt != null
          ? Uri.parse('$_baseUrl/events?since=$_lastCapturedAt')
          : Uri.parse('$_baseUrl/events');

      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['ok'] == true) {
          final List<dynamic> eventsJson = data['events'];
          if (eventsJson.isEmpty) return;

          final newEvents =
              eventsJson.map((json) => Event.fromJson(json)).toList();

          _lastCapturedAt = newEvents.first.capturedAt;

          if (state.isEmpty) {
            state = newEvents;
          } else {
            state = [...newEvents, ...state];
          }
        }
      }
    } catch (e) {
      // 무시
    }
  }

  /// 더미 이벤트 생성
  Future<void> generateMockEvent() async {
    try {
      final uri = Uri.parse('$_baseUrl/test/create_event');

      await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "place_id": "00_test",
          "anomaly_flag": 1,
          "summary_text": "[테스트] 카메라 강제 진동 감지"
        }),
      );

      await _fetchEvents();
    } catch (e) {
      // 무시
    }
  }

  /// 실제 query 이미지를 서버로 보내서 이상감지 이벤트 생성
  Future<void> sendQueryBatch({
    required String placeId,
    required List<({String filename, List<int> bytes})> images,
    String? label,
  }) async {
    try {
      final uri = Uri.parse('$_baseUrl/place_imgs');

      final meta = {
        "place_id": placeId,
        "timestamp": DateTime.now().toIso8601String(),
        "n_frames": images.length,
        "mode": "query",
        "label": label,
      };

      final request = http.MultipartRequest('POST', uri);
      request.fields['meta'] = jsonEncode(meta);

      for (final img in images) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'images',
            img.bytes,
            filename: img.filename,
          ),
        );
      }

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['ok'] == true) {
          await _fetchEvents();
        } else {
          print('sendQueryBatch ok=false: ${response.body}');
        }
      } else {
        print('sendQueryBatch failed: ${response.statusCode} ${response.body}');
      }
    } catch (e) {
      print('sendQueryBatch error: $e');
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    _pollingTimer?.cancel();
    super.dispose();
  }
}

// Event 상태를 전역으로 제공하는 Provider 선언부
final eventListProvider =
    StateNotifierProvider<EventListNotifier, List<Event>>((ref) {
  final notifier = EventListNotifier();
  notifier.startPolling();
  return notifier;
});

final selectedEventProvider = StateProvider<Event?>((ref) => null);