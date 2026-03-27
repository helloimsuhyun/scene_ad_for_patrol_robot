class RobotPose {
  final double? x;
  final double? y;
  final double? yaw;
  final String status;
  final String? timestamp;

  RobotPose({
    this.x,
    this.y,
    this.yaw,
    required this.status,
    this.timestamp,
  });

  factory RobotPose.fromJson(Map<String, dynamic> json) {
    return RobotPose(
      x: json['x']?.toDouble(),
      y: json['y']?.toDouble(),
      yaw: json['yaw']?.toDouble(),
      status: json['status'] ?? 'idle',
      timestamp: json['timestamp'],
    );
  }
}

class RobotGoal {
  final double? x;
  final double? y;
  final double? yaw;
  final String? nextPlaceId;
  final String? timestamp;

  RobotGoal({
    this.x,
    this.y,
    this.yaw,
    this.nextPlaceId,
    this.timestamp,
  });

  factory RobotGoal.fromJson(Map<String, dynamic> json) {
    return RobotGoal(
      x: json['x']?.toDouble(),
      y: json['y']?.toDouble(),
      yaw: json['yaw']?.toDouble(),
      nextPlaceId: json['next_place_id'],
      timestamp: json['timestamp'],
    );
  }
}
