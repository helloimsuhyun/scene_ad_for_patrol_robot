import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import '../models/robot_state.dart';
import 'server_config_provider.dart';

class RobotPoseNotifier extends StateNotifier<RobotPose?> {
  final Ref ref;
  Timer? _timer;

  RobotPoseNotifier(this.ref) : super(null) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      fetchRobotPose();
    });
    fetchRobotPose(); // initial fetch
  }

  Future<void> fetchRobotPose() async {
    final config = ref.read(serverConfigProvider);
    try {
      final response = await http.get(Uri.parse('${config.baseUrl}/robot/pose'));
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
  return RobotPoseNotifier(ref);
});

class RobotGoalNotifier extends StateNotifier<RobotGoal?> {
  final Ref ref;
  Timer? _timer;

  RobotGoalNotifier(this.ref) : super(null) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      fetchRobotGoal();
    });
    fetchRobotGoal();
  }

  Future<void> fetchRobotGoal() async {
    final config = ref.read(serverConfigProvider);
    try {
      final response = await http.get(Uri.parse('${config.baseUrl}/robot/goal'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['ok'] == true && data['goal'] != null) {
          state = RobotGoal.fromJson(data['goal']);
        }
      }
    } catch (e) {
      // ignore
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final robotGoalProvider = StateNotifierProvider<RobotGoalNotifier, RobotGoal?>((ref) {
  return RobotGoalNotifier(ref);
});

class YoloModeNotifier extends StateNotifier<int> {
  final Ref ref;
  Timer? _timer;

  YoloModeNotifier(this.ref) : super(0) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) {
      fetchYoloMode();
    });
    fetchYoloMode();
  }

  Future<void> fetchYoloMode() async {
    final config = ref.read(serverConfigProvider);
    try {
      final response = await http.get(Uri.parse('${config.baseUrl}/robot/yolo_mode'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['ok'] == true) {
          final int newVal = (data['yolo_mode'] as num).toInt();
          if (state != newVal) {
            state = newVal;
          }
        }
      }
    } catch (e) {
      // ignore
    }
  }

  Future<void> setMode(int mode) async {
    final config = ref.read(serverConfigProvider);
    try {
      // Optimistic update
      state = mode;
      final response = await http.patch(
        Uri.parse('${config.baseUrl}/robot/yolo_mode'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'yolo_mode': mode}),
      );
      if (response.statusCode != 200) {
        // Revert on failure
        fetchYoloMode();
      }
    } catch (e) {
      fetchYoloMode();
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final yoloModeProvider = StateNotifierProvider<YoloModeNotifier, int>((ref) {
  return YoloModeNotifier(ref);
});

final yoloDrawingModeProvider = StateProvider<bool>((ref) => false);

final yoloShowRegionsProvider = StateProvider<bool>((ref) => true);

class RobotBatteryNotifier extends StateNotifier<int> {
  final Ref ref;
  Timer? _timer;

  RobotBatteryNotifier(this.ref) : super(100) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 10), (timer) {
      fetchBattery();
    });
    fetchBattery();
  }

  Future<void> fetchBattery() async {
    final config = ref.read(serverConfigProvider);
    try {
      final response = await http.get(Uri.parse('${config.baseUrl}/robot/battery'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['battery'] != null) {
          state = (data['battery'] as num).toInt();
        }
      }
    } catch (e) {
      // ignore
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final robotBatteryProvider = StateNotifierProvider<RobotBatteryNotifier, int>((ref) {
  return RobotBatteryNotifier(ref);
});
