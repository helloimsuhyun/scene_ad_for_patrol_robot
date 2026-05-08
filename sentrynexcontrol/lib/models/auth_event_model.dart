class AuthEvent {
  final String authEventId;
  final String? trackingPersonId;
  final String? yoloEventId;
  final String? employeeId;
  final String timestamp;
  final String status;
  final String? rfidUid;
  final String? employeeName;
  final String? resultMessage;
  final String? imageUrl; // image_url from API
  final int? sourceRegionId;
  final String? sourceRegionName;
  final double? x;
  final double? y;
  final double? yaw;
  final int adminChecked;
  final String? adminLabel;
  final String createdAt;

  AuthEvent({
    required this.authEventId,
    this.trackingPersonId,
    this.yoloEventId,
    this.employeeId,
    required this.timestamp,
    required this.status,
    this.rfidUid,
    this.employeeName,
    this.resultMessage,
    this.imageUrl,
    this.sourceRegionId,
    this.sourceRegionName,
    this.x,
    this.y,
    this.yaw,
    required this.adminChecked,
    this.adminLabel,
    required this.createdAt,
  });

  factory AuthEvent.fromJson(Map<String, dynamic> json) {
    return AuthEvent(
      authEventId: json['auth_event_id'] ?? '',
      trackingPersonId: json['tracking_person_id'],
      yoloEventId: json['yolo_event_id'],
      employeeId: json['employee_id'],
      timestamp: json['timestamp'] ?? '',
      status: json['status'] ?? 'waiting_rfid',
      rfidUid: json['rfid_uid'],
      employeeName: json['employee_name'],
      resultMessage: json['result_message'],
      imageUrl: json['image_url'], // API returns 'image_url' mapped from 'image_path'
      sourceRegionId: json['source_region_id'],
      sourceRegionName: json['source_region_name'],
      x: json['x']?.toDouble(),
      y: json['y']?.toDouble(),
      yaw: json['yaw']?.toDouble(),
      adminChecked: json['admin_checked'] ?? 0,
      adminLabel: json['admin_label'],
      createdAt: json['created_at'] ?? '',
    );
  }
}
