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

  AudioEvent copyWith({
    String? audioEventId,
    String? timestamp,
    String? audioPath,
    String? audioUrl,
    double? x,
    double? y,
    double? yaw,
    double? doa,
    String? modelLabel,
    String? adminLabel,
    int? adminChecked,
  }) {
    return AudioEvent(
      audioEventId: audioEventId ?? this.audioEventId,
      timestamp: timestamp ?? this.timestamp,
      audioPath: audioPath ?? this.audioPath,
      audioUrl: audioUrl ?? this.audioUrl,
      x: x ?? this.x,
      y: y ?? this.y,
      yaw: yaw ?? this.yaw,
      doa: doa ?? this.doa,
      modelLabel: modelLabel ?? this.modelLabel,
      adminLabel: adminLabel ?? this.adminLabel,
      adminChecked: adminChecked ?? this.adminChecked,
    );
  }
}
