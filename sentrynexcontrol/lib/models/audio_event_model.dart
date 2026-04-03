class AudioEvent {
  final String audioEventId;
  final String timestamp;
  final String audioPath;
  final String audioUrl;
  final double? x;
  final double? y;
  final double? yaw;
  final double? doa;
  final String? modelLabel;
  final String? adminLabel;
  final int adminChecked;

  AudioEvent({
    required this.audioEventId,
    required this.timestamp,
    required this.audioPath,
    required this.audioUrl,
    this.x,
    this.y,
    this.yaw,
    this.doa,
    this.modelLabel,
    this.adminLabel,
    this.adminChecked = 0,
  });

  factory AudioEvent.fromJson(Map<String, dynamic> json) {
    return AudioEvent(
      audioEventId: json['audio_event_id'] ?? '',
      timestamp: json['timestamp'] ?? '',
      audioPath: json['audio_path'] ?? '',
      audioUrl: json['audio_url'] ?? '',
      x: json['x'] != null ? (json['x'] as num).toDouble() : null,
      y: json['y'] != null ? (json['y'] as num).toDouble() : null,
      yaw: json['yaw'] != null ? (json['yaw'] as num).toDouble() : null,
      doa: json['doa'] != null ? (json['doa'] as num).toDouble() : null,
      modelLabel: json['model_label'],
      adminLabel: json['admin_label'],
      adminChecked: json['admin_checked'] ?? 0,
    );
  }
}
