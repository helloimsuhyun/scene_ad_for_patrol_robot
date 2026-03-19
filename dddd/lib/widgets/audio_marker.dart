// audio_marker.dart
// 지도(DataCenterMap)에 표시되는 오디오 이벤트 마커 위젯
//
// ── 라벨링 방식 ──────────────────────────────────────────
//  [빠른 확인]   : "✓ 확인" 버튼 → 'confirmed' 라벨 저장
//  [상세 라벨링] : 카테고리 목록 클릭 → 해당 라벨 저장
//
// ── 마커 색상 규칙 ────────────────────────────────────────
//  빨강  : 위험 소음 (비명, 충격음, 유리깨짐, 경보음, 다툼)
//  주황  : 미분류 / 기타
//  회색  : 오탐으로 확인됨 (라벨 저장 완료 → 사라짐)

import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:audioplayers/audioplayers.dart';
import '../models/audio_event_model.dart';
import '../providers/audio_event_provider.dart';

// ── 라벨 정의 ─────────────────────────────────────────────────────────────────
// isDangerous: 마커를 빨간색으로 표시할 라벨 목록
// (오탐/무음 등 위험하지 않은 건 false)
class _LabelOption {
  final String value;        // 서버에 저장되는 실제 값
  final String display;      // UI에 표시되는 텍스트
  final IconData icon;
  final bool isDangerous;    // true면 빨간 마커로 바꿔서 표시

  const _LabelOption({
    required this.value,
    required this.display,
    required this.icon,
    required this.isDangerous,
  });
}

const List<_LabelOption> _labelOptions = [
  _LabelOption(value: 'scream',      display: '비명',       icon: Icons.record_voice_over, isDangerous: true),
  _LabelOption(value: 'impact',      display: '충격음',     icon: Icons.bolt,              isDangerous: true),
  _LabelOption(value: 'glass_break', display: '유리깨짐',   icon: Icons.broken_image,      isDangerous: true),
  _LabelOption(value: 'alarm',       display: '경보음',     icon: Icons.crisis_alert,      isDangerous: true),
  _LabelOption(value: 'argument',    display: '다툼/고성',  icon: Icons.people,            isDangerous: true),
  _LabelOption(value: 'gunshot',     display: '총성/폭발음',icon: Icons.warning_amber,     isDangerous: true),
  _LabelOption(value: 'confirmed',   display: '✓ 일반 확인',icon: Icons.check_circle,      isDangerous: false),
  _LabelOption(value: 'false_alarm', display: '오탐',       icon: Icons.do_not_disturb,    isDangerous: false),
  _LabelOption(value: 'noise',       display: '소음(무해)', icon: Icons.volume_mute,        isDangerous: false),
];

// 위험 라벨 값 집합 (마커 색상 판단용)
final _dangerousLabels = _labelOptions
    .where((l) => l.isDangerous)
    .map((l) => l.value)
    .toSet();

// ──────────────────────────────────────────────────────────────────────────────

// 현재 팝업이 열린 마커 ID를 전역 관리 (다른 마커 탭하면 이전 팝업 자동 닫힘)
final _selectedAudioEventProvider = StateProvider<String?>((ref) => null);

class AudioMarker extends ConsumerStatefulWidget {
  final AudioEventData event;
  const AudioMarker({super.key, required this.event});

  @override
  ConsumerState<AudioMarker> createState() => _AudioMarkerState();
}

class _AudioMarkerState extends ConsumerState<AudioMarker> {
  final AudioPlayer _player = AudioPlayer();
  PlayerState _playerState = PlayerState.stopped;
  StreamSubscription<PlayerState>? _playerStateSub;

  static const String _baseUrl = 'http://localhost:8000';

  // 상세 라벨 패널 표시 여부
  bool _showLabelPanel = false;

  @override
  void initState() {
    super.initState();
    _playerStateSub = _player.onPlayerStateChanged.listen((s) {
      if (mounted) setState(() => _playerState = s);
    });
  }

  @override
  void dispose() {
    _playerStateSub?.cancel();
    _player.dispose();
    super.dispose();
  }

  // ── 재생/일시정지 ──────────────────────────────────────────────────────────
  Future<void> _togglePlay() async {
    if (_playerState == PlayerState.playing) {
      await _player.pause();
    } else {
      await _player.play(UrlSource('$_baseUrl${widget.event.audioUrl}'));
    }
  }

  // ── 라벨 저장 (서버 PATCH → 로컬 목록에서 제거) ────────────────────────────
  Future<void> _applyLabel(String labelValue) async {
    // 팝업 즉시 닫기
    ref.read(_selectedAudioEventProvider.notifier).state = null;
    await ref.read(audioEventListProvider.notifier).applyAdminLabel(
      audioEventId: widget.event.audioEventId,
      adminLabel: labelValue,
    );
  }

  // ── model_label 또는 null 기반으로 마커가 위험 소음인지 판단 ────────────────
  bool get _isDangerous =>
      widget.event.modelLabel != null &&
      _dangerousLabels.contains(widget.event.modelLabel);

  @override
  Widget build(BuildContext context) {
    print('DEBUG: Building AudioMarker for ${widget.event.audioEventId} at (${widget.event.x}, ${widget.event.y})');
    final selectedId = ref.watch(_selectedAudioEventProvider);
    final isSelected = selectedId == widget.event.audioEventId;
    final isPlaying = _playerState == PlayerState.playing;

    // 위험 여부에 따라 마커 색상 분기
    final markerColor = _isDangerous ? const Color(0xFFEF4444) : const Color(0xFFFF8C00);
    final glowColor = _isDangerous
        ? const Color(0x99EF4444)
        : const Color(0x99FF8C00);

    final alignment = _xyToAlignment(widget.event.x, widget.event.y);

    return Align(
      alignment: alignment,
      child: SizedBox(
        width: 260,
        height: 360,
        child: Stack(
          clipBehavior: Clip.none,
          alignment: Alignment.center,
          children: [
            //---------- 팝업 (마커 선택 시 마커 위에 표시) ----------
            if (isSelected)
              Positioned(
                bottom: 180 + 10,
                child: _buildPopup(isPlaying, markerColor),
              ),

            //---------- 마커 본체 ----------
            GestureDetector(
              onTap: () {
                // 다른 마커 탭하면 상세 패널도 초기화
                setState(() => _showLabelPanel = false);
                final notifier = ref.read(_selectedAudioEventProvider.notifier);
                notifier.state = (notifier.state == widget.event.audioEventId)
                    ? null
                    : widget.event.audioEventId;
              },
              child: Container(
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: markerColor,
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 2),
                  boxShadow: [
                    BoxShadow(color: glowColor, blurRadius: 14, spreadRadius: 4),
                  ],
                ),
                child: Icon(
                  // 위험 소음이면 경고 아이콘, 아니면 스피커
                  _isDangerous ? Icons.warning_rounded : Icons.volume_up,
                  color: Colors.white,
                  size: 18,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── 팝업 전체 ──────────────────────────────────────────────────────────────
  Widget _buildPopup(bool isPlaying, Color accentColor) {
    return Container(
      width: 248,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1C1E2B),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: accentColor, width: 1.5),
        boxShadow: const [
          BoxShadow(color: Colors.black54, blurRadius: 16, offset: Offset(0, 8)),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 팝업 헤더 ----------
          Row(
            children: [
              Icon(Icons.hearing, color: accentColor, size: 14),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  _isDangerous ? '⚠ 위험 소음 감지' : '오디오 이벤트 감지',
                  style: TextStyle(
                    color: accentColor,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const Divider(color: Color(0xFF2D3041), height: 14),

          //---------- 정보 행 ----------
          _infoRow('라벨',   widget.event.modelLabel ?? '미분류'),
          const SizedBox(height: 3),
          _infoRow('좌표',   'x=${widget.event.x.toStringAsFixed(2)}, y=${widget.event.y.toStringAsFixed(2)}'),
          const SizedBox(height: 3),
          _infoRow('방향',   '${widget.event.doa.toStringAsFixed(1)}°'),
          const SizedBox(height: 10),

          //---------- 재생 버튼 ----------
          GestureDetector(
            onTap: _togglePlay,
            child: Container(
              height: 34,
              decoration: BoxDecoration(
                color: isPlaying
                    ? const Color(0xFF374151)
                    : accentColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: accentColor, width: 1),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    isPlaying ? Icons.pause : Icons.play_arrow,
                    color: accentColor,
                    size: 16,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    isPlaying ? '일시정지' : '녹음 재생',
                    style: TextStyle(color: accentColor, fontSize: 12),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),

          //---------- 라벨링 버튼 영역 ----------
          if (!_showLabelPanel)
            _buildQuickActionRow()
          else
            _buildLabelGrid(),
        ],
      ),
    );
  }

  // ── 빠른 액션 행 (✓ 확인 + 상세 라벨링 버튼) ─────────────────────────────
  Widget _buildQuickActionRow() {
    return Row(
      children: [
        //---------- ✓ 빠른 확인 버튼 ----------
        Expanded(
          child: GestureDetector(
            onTap: () => _applyLabel('confirmed'),
            child: Container(
              height: 34,
              decoration: BoxDecoration(
                color: const Color(0xFF22C55E).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF22C55E), width: 1),
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.check, color: Color(0xFF22C55E), size: 16),
                  SizedBox(width: 4),
                  Text('확인', style: TextStyle(color: Color(0xFF22C55E), fontSize: 12)),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),

        //---------- 상세 라벨링 버튼 ----------
        Expanded(
          child: GestureDetector(
            onTap: () => setState(() => _showLabelPanel = true),
            child: Container(
              height: 34,
              decoration: BoxDecoration(
                color: const Color(0xFF6366F1).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF6366F1), width: 1),
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.label_outline, color: Color(0xFF6366F1), size: 16),
                  SizedBox(width: 4),
                  Text('상세 라벨링', style: TextStyle(color: Color(0xFF6366F1), fontSize: 12)),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ── 상세 라벨 선택 그리드 ─────────────────────────────────────────────────
  Widget _buildLabelGrid() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        //---------- 뒤로 가기 ----------
        GestureDetector(
          onTap: () => setState(() => _showLabelPanel = false),
          child: const Row(
            children: [
              Icon(Icons.arrow_back_ios, size: 11, color: Colors.white38),
              Text('라벨 선택', style: TextStyle(color: Colors.white38, fontSize: 11)),
            ],
          ),
        ),
        const SizedBox(height: 8),

        //---------- 라벨 버튼 그리드 ----------
        Wrap(
          spacing: 6,
          runSpacing: 6,
          children: _labelOptions.map((opt) {
            // 위험 vs 비위험 색상 분기
            final color = opt.isDangerous
                ? const Color(0xFFEF4444)
                : const Color(0xFF6B7280);
            return GestureDetector(
              onTap: () => _applyLabel(opt.value),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: color.withValues(alpha: 0.6), width: 1),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(opt.icon, size: 12, color: color),
                    const SizedBox(width: 4),
                    Text(opt.display, style: TextStyle(color: color, fontSize: 11)),
                  ],
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  // ── 정보 행 헬퍼 ──────────────────────────────────────────────────────────
  Widget _infoRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 38,
          child: Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ),
        Expanded(
          child: Text(value, style: const TextStyle(color: Colors.white70, fontSize: 11)),
        ),
      ],
    );
  }

  // x,y 좌표 -> Alignment 변환
  // 실제 환경에 맞게 _coordScale 값을 조정하세요 (현재: ±5m 기준)
  // 좌표가 없거나 의미 없는 값(NaN, Infinite, 0,0)이면
  // audioEventId를 시드로 쓴 결정적 랜덤 위치를 반환 (매 빌드마다 같은 자리)
  static const double _coordScale = 5.0;

  Alignment _xyToAlignment(double x, double y) {
    // 유효하지 않은 좌표 판단:
    //  - NaN / Infinite
    //  - (0,0) → 로봇이 실제 좌표를 보내지 않은 경우로 간주
    final isInvalid = x.isNaN || y.isNaN || x.isInfinite || y.isInfinite ||
        (x == 0.0 && y == 0.0);

    if (isInvalid) {
      // audioEventId의 hashCode를 시드로 사용 → 같은 이벤트는 항상 같은 위치
      final rng = Random(widget.event.audioEventId.hashCode);
      // 엣지에 몰리지 않도록 ±0.75 범위 내에서만 배치
      final rx = (rng.nextDouble() * 1.5) - 0.75;
      final ry = (rng.nextDouble() * 1.5) - 0.75;
      return Alignment(rx, ry);
    }

    final ax = (x / _coordScale).clamp(-1.0, 1.0);
    final ay = (y / _coordScale).clamp(-1.0, 1.0);
    return Alignment(ax, ay);
  }
}
