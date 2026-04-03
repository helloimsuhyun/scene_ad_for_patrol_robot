import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

final String baseUrl = 'http://127.0.0.1:8000';

final placesProvider = FutureProvider.autoDispose<Map<String, dynamic>>((ref) async {
  final response = await http.get(Uri.parse('$baseUrl/places'));
  if (response.statusCode == 200) {
    return jsonDecode(response.body);
  } else {
    throw Exception('Failed to load places');
  }
});

// 순찰 실행 중 여부를 관리하는 전역 상태 프로바이더
final patrolStatusProvider = StateProvider<bool>((ref) => false);


class ControlActions {
  static Future<void> recalibrateAll(WidgetRef ref) async {
    final response = await http.post(Uri.parse('$baseUrl/places/recalibrate_all'));
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> updateDisplayName(WidgetRef ref, String placeId, String name) async {
    final response = await http.patch(
      Uri.parse('$baseUrl/places/$placeId/display_name'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'display_name': name}),
    );
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> updatePatrolEnabled(WidgetRef ref, String placeId, bool enabled) async {
    final response = await http.patch(
      Uri.parse('$baseUrl/places/$placeId/patrol_enabled'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'patrol_enabled': enabled}),
    );
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> reorderPatrol(WidgetRef ref, List<String> placeIds) async {
    final response = await http.patch(
      Uri.parse('$baseUrl/places/patrol_order'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'place_ids': placeIds}),
    );
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }


  static Future<void> setMode(WidgetRef ref, String placeId, String mode) async {
    final response = await http.post(
      Uri.parse('$baseUrl/places/$placeId/config'),
      body: {'mode': mode},
    );
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> deletePlace(WidgetRef ref, String placeId) async {
    final response = await http.delete(Uri.parse('$baseUrl/places/$placeId'));
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> deleteThreshold(WidgetRef ref, String placeId) async {
    final response = await http.delete(Uri.parse('$baseUrl/places/$placeId/threshold'));
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> deleteAllPlaces(WidgetRef ref) async {
    final response = await http.delete(Uri.parse('$baseUrl/places'));
    if (response.statusCode == 200) {
      ref.invalidate(placesProvider);
    }
  }

  static Future<void> applyRouteAndStart(WidgetRef ref, List<Map<String, dynamic>> allPlaces, List<String> selectedPlaceIds) async {
    // 1. Update patrol_enabled
    for (final p in allPlaces) {
      final pid = p['place_id'].toString();
      final enabled = selectedPlaceIds.contains(pid);
      
      // Optimization
      final currentEnabled = p['patrol_enabled'] == 1 || p['patrol_enabled'] == true;
      if (currentEnabled != enabled) {
        await http.patch(
          Uri.parse('$baseUrl/places/$pid/patrol_enabled'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'patrol_enabled': enabled}),
        );
      }
    }
    
    // 2. Update patrol_order
    final unselected = allPlaces
        .map((e) => e['place_id'].toString())
        .where((id) => !selectedPlaceIds.contains(id))
        .toList();
    final newOrder = [...selectedPlaceIds, ...unselected];
    
    await http.patch(
      Uri.parse('$baseUrl/places/patrol_order'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'place_ids': newOrder}),
    );
    
    // 3. Start patrol
    await http.post(
      Uri.parse('http://127.0.0.1:8000/robot/command'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'command': 'start_patrol'}),
    );
    
    // 순찰 상태 전역 반영
    ref.read(patrolStatusProvider.notifier).state = true;
    ref.invalidate(placesProvider);
  }
}
