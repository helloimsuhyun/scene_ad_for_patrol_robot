class Frame {
  final String frameId;
  final String eventId;
  final int idx;
  final String imagePath;
  final double? frameScore;
  final String? captureTime;

  Frame({
    required this.frameId,
    required this.eventId,
    required this.idx,
    required this.imagePath,
    this.frameScore,
    this.captureTime,
  });

  factory Frame.fromJson(Map<String, dynamic> json) {
    return Frame(
      frameId: json['frame_id'] ?? '',
      eventId: json['event_id'] ?? '',
      idx: json['idx'] ?? 0,
      imagePath: json['image_path'] ?? '',
      frameScore: json['frame_score']?.toDouble(),
      captureTime: json['capture_time'],
    );
  }
}

class Event {
  final String eventId;
  final String placeId;
  final String capturedAt;
  final int anomalyFlag;
  final double? anomalyScore;
  final double? thresholdUsed;
  final String? refBankId;
  final String? refTopkJson;
  final String? summaryText;
  final String? createdAt;
  final int? adminChecked;
  final String? adminLabel;
  final double? x;
  final double? y;
  final double? yaw;
  final String? verifiedChangeImageUrl;
  
  // 백엔드에서 조인해서 보내줄 경우를 위한 frame 리스트
  final List<Frame> frames;

  Event({
    required this.eventId,
    required this.placeId,
    required this.capturedAt,
    required this.anomalyFlag,
    this.anomalyScore,
    this.thresholdUsed,
    this.refBankId,
    this.refTopkJson,
    this.summaryText,
    this.createdAt,
    this.adminChecked,
    this.adminLabel,
    this.x,
    this.y,
    this.yaw,
    this.verifiedChangeImageUrl,
    this.frames = const [],
  });

  factory Event.fromJson(Map<String, dynamic> json) {
    var framesList = json['frames'] as List?;
    List<Frame> parsedFrames = framesList != null 
        ? framesList.map((f) => Frame.fromJson(f)).toList() 
        : [];

    return Event(
      eventId: json['event_id'] ?? '',
      placeId: json['place_id'] ?? '',
      capturedAt: json['captured_at'] ?? '',
      anomalyFlag: json['anomaly_flag'] ?? 0,
      anomalyScore: json['anomaly_score']?.toDouble(),
      thresholdUsed: json['threshold_used']?.toDouble(),
      refBankId: json['ref_bank_id'],
      refTopkJson: json['ref_topk_json'],
      summaryText: json['summary_text'],
      createdAt: json['created_at'],
      adminChecked: json['admin_checked'],
      adminLabel: json['admin_label'],
      x: json['x']?.toDouble(),
      y: json['y']?.toDouble(),
      yaw: json['yaw']?.toDouble(),
      verifiedChangeImageUrl: json['verified_change_image_url'],
      frames: parsedFrames,
    );
  }
}
