// audio_event_model.dart
// 서버의 /audio_events 응답 JSON을 Dart 객체로 변환하는 모델

class AudioEventData {
  final String audioEventId;
  final String timestamp;
  final String audioPath;
  final String audioUrl;  // /audio/... 형태의 상대경로
  final double x;
  final double y;
  final double yaw;
  final double doa;
  final String? modelLabel;   // 모델이 판정한 라벨
  final String? adminLabel;   // 관리자가 확인한 라벨 (있으면 이미 처리 완료)
  final String createdAt;

  const AudioEventData({
    required this.audioEventId,
    required this.timestamp,
    required this.audioPath,
    required this.audioUrl,
    required this.x,
    required this.y,
    required this.yaw,
    required this.doa,
    this.modelLabel,
    this.adminLabel,
    required this.createdAt,
  });

  /// 서버 JSON -> AudioEvent 변환
  factory AudioEventData.fromJson(Map<String, dynamic> json) {
    return AudioEventData(
      audioEventId: json['audio_event_id'] as String,
      timestamp: json['timestamp'] as String,
      audioPath: json['audio_path'] as String,
      // audio_url이 없을 경우 audio_path에서 파일명만 추출해서 사용
      audioUrl: json['audio_url'] as String? ?? '/audio/${(json['audio_path'] as String).split('/').last}',
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      yaw: (json['yaw'] as num).toDouble(),
      doa: (json['doa'] as num).toDouble(),
      modelLabel: json['model_label'] as String?,
      adminLabel: json['admin_label'] as String?,
      createdAt: json['created_at'] as String,
    );
  }
}
