import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/robot_state.dart';

class RobotPoseNotifier extends StateNotifier<RobotPose?> {
  Timer? _timer;

  RobotPoseNotifier() : super(null) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      fetchRobotPose();
    });
    fetchRobotPose(); // initial fetch
  }

  Future<void> fetchRobotPose() async {
    try {
      final response = await http.get(Uri.parse('http://127.0.0.1:8000/robot/pose'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['ok'] == true && data['pose'] != null) {
          state = RobotPose.fromJson(data['pose']);
        }
      }
    } catch (e) {
      // ignore silently to prevent flood of logs during disconnects
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final robotPoseProvider = StateNotifierProvider<RobotPoseNotifier, RobotPose?>((ref) {
  return RobotPoseNotifier();
});
