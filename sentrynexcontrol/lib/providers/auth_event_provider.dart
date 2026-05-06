import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/auth_event_model.dart';
import 'server_config_provider.dart';

final authEventListProvider = StateNotifierProvider<AuthEventNotifier, List<AuthEvent>>((ref) {
  final baseUrl = ref.watch(serverConfigProvider).baseUrl;
  return AuthEventNotifier(baseUrl);
});

class AuthEventNotifier extends StateNotifier<List<AuthEvent>> {
  final String baseUrl;
  Timer? _timer;

  AuthEventNotifier(this.baseUrl) : super([]) {
    _startPolling();
  }

  void _startPolling() {
    _fetchEvents();
    _timer = Timer.periodic(const Duration(seconds: 3), (_) {
      _fetchEvents();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetchEvents() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/auth_events?limit=50'));
      if (response.statusCode == 200) {
        final data = json.decode(utf8.decode(response.bodyBytes));
        if (data['ok'] == true) {
          final List<dynamic> eventsJson = data['auth_events'];
          final events = eventsJson.map((json) => AuthEvent.fromJson(json)).toList();
          
          // 상태가 변경되었을 때만 갱신
          if (mounted) {
            state = events;
          }
        }
      }
    } catch (e) {
      debugPrint('AuthEvent fetch error: $e');
    }
  }

  Future<void> updateAdminLabel(String authEventId, String label) async {
    try {
      final response = await http.patch(
        Uri.parse('$baseUrl/auth_events/$authEventId/label'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'admin_label': label}),
      );

      if (response.statusCode == 200) {
        // 즉각적인 로컬 상태 반영 (UI 응답성 개선)
        state = state.map((event) {
          if (event.authEventId == authEventId) {
            return AuthEvent(
              authEventId: event.authEventId,
              trackingPersonId: event.trackingPersonId,
              yoloEventId: event.yoloEventId,
              employeeId: event.employeeId,
              timestamp: event.timestamp,
              status: event.status,
              rfidUid: event.rfidUid,
              employeeName: event.employeeName,
              resultMessage: event.resultMessage,
              imageUrl: event.imageUrl,
              sourceRegionId: event.sourceRegionId,
              sourceRegionName: event.sourceRegionName,
              x: event.x,
              y: event.y,
              yaw: event.yaw,
              adminChecked: 1, // 로컬 갱신
              adminLabel: label,
              createdAt: event.createdAt,
            );
          }
          return event;
        }).toList();
        
        // 백엔드 폴링은 계속되므로 곧 동기화됨
      }
    } catch (e) {
      debugPrint('AuthEvent update label error: $e');
    }
  }
}
