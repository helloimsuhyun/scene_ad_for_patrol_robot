class YoloEvent {
  final String yoloEventId;
  final String timestamp;
  final String? imagePath;
  final String? imageUrl;
  final double? x;
  final double? y;
  final double? yaw;
  final int personCount;
  final String? eventType;
  final int? sourceRegionId;
  final String? sourceRegionName;
  final double? dwellTimeSec;
  final int adminChecked;
  final String? adminLabel;
  final String? createdAt;

  YoloEvent({
    required this.yoloEventId,
    required this.timestamp,
    this.imagePath,
    this.imageUrl,
    this.x,
    this.y,
    this.yaw,
    this.personCount = 0,
    this.eventType,
    this.sourceRegionId,
    this.sourceRegionName,
    this.dwellTimeSec,
    this.adminChecked = 0,
    this.adminLabel,
    this.createdAt,
  });

  factory YoloEvent.fromJson(Map<String, dynamic> json) {
    return YoloEvent(
      yoloEventId: json['yolo_event_id'] ?? '',
      timestamp: json['timestamp'] ?? '',
      imagePath: json['image_path'],
      imageUrl: json['image_url'],
      x: json['x']?.toDouble(),
      y: json['y']?.toDouble(),
      yaw: json['yaw']?.toDouble(),
      personCount: json['person_count'] ?? 0,
      eventType: json['event_type'],
      sourceRegionId: json['source_region_id'],
      sourceRegionName: json['source_region_name'],
      dwellTimeSec: json['dwell_time_sec']?.toDouble(),
      adminChecked: json['admin_checked'] ?? 0,
      adminLabel: json['admin_label'],
      createdAt: json['created_at'],
    );
  }

  YoloEvent copyWith({
    String? yoloEventId,
    String? timestamp,
    String? imagePath,
    String? imageUrl,
    double? x,
    double? y,
    double? yaw,
    int? personCount,
    String? eventType,
    int? sourceRegionId,
    String? sourceRegionName,
    double? dwellTimeSec,
    int? adminChecked,
    String? adminLabel,
    String? createdAt,
  }) {
    return YoloEvent(
      yoloEventId: yoloEventId ?? this.yoloEventId,
      timestamp: timestamp ?? this.timestamp,
      imagePath: imagePath ?? this.imagePath,
      imageUrl: imageUrl ?? this.imageUrl,
      x: x ?? this.x,
      y: y ?? this.y,
      yaw: yaw ?? this.yaw,
      personCount: personCount ?? this.personCount,
      eventType: eventType ?? this.eventType,
      sourceRegionId: sourceRegionId ?? this.sourceRegionId,
      sourceRegionName: sourceRegionName ?? this.sourceRegionName,
      dwellTimeSec: dwellTimeSec ?? this.dwellTimeSec,
      adminChecked: adminChecked ?? this.adminChecked,
      adminLabel: adminLabel ?? this.adminLabel,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
