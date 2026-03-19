import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

final String baseUrl = 'http://127.0.0.1:8000';

class CalibrationStatus {
  final bool globalCalibrating;
  final int total;
  final int done;
  final String? currentPlaceId;

  CalibrationStatus({
    required this.globalCalibrating,
    required this.total,
    required this.done,
    this.currentPlaceId,
  });

  factory CalibrationStatus.fromJson(Map<String, dynamic> json) {
    final prog = json['calib_progress'] ?? {};
    return CalibrationStatus(
      globalCalibrating: json['global_calibrating'] ?? false,
      total: prog['total'] ?? 0,
      done: prog['done'] ?? 0,
      currentPlaceId: prog['current_place_id'],
    );
  }
}

class CalibrationNotifier extends StateNotifier<CalibrationStatus> {
  Timer? _timer;

  CalibrationNotifier() : super(CalibrationStatus(globalCalibrating: false, total: 0, done: 0)) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 1), (_) async {
      try {
        final response = await http.get(Uri.parse('$baseUrl/calibration_status'));
        if (response.statusCode == 200) {
          state = CalibrationStatus.fromJson(jsonDecode(response.body));
        }
      } catch (e) {
        // Handle polling error silently
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final calibrationProvider = StateNotifierProvider<CalibrationNotifier, CalibrationStatus>((ref) {
  return CalibrationNotifier();
});
