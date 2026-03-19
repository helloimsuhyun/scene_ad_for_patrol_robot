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

class ControlActions {
  static Future<void> recalibrateAll(WidgetRef ref) async {
    final response = await http.post(Uri.parse('$baseUrl/places/recalibrate_all'));
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
}
