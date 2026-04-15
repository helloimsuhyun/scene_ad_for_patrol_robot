import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:audioplayers/audioplayers.dart' hide AudioEvent;
import 'package:http/http.dart' as http;
import '../providers/audio_provider.dart';
import '../models/audio_event_model.dart';
import '../providers/server_config_provider.dart';

class AudioEventList extends ConsumerWidget {
  final bool showOnlyUnchecked;
  const AudioEventList({super.key, this.showOnlyUnchecked = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Riverpod 상태에서 이벤트 목록 가져오기
    final allEvents = ref.watch(audioEventListProvider);
    
    // 조건에 따른 필터링 (showOnlyUnchecked 가 true 일 때만 필터링)
    final audioEvents = List<AudioEvent>.from(showOnlyUnchecked
        ? allEvents.where((e) {
            final bool isChecked = (e.adminChecked == 1) || (e.adminLabel != null && e.adminLabel!.isNotEmpty);
            return !isChecked;
          })
        : allEvents);

    audioEvents.sort((a, b) {
      bool aChecked = (a.adminChecked == 1) || (a.adminLabel != null && a.adminLabel!.isNotEmpty);
      bool bChecked = (b.adminChecked == 1) || (b.adminLabel != null && b.adminLabel!.isNotEmpty);
      
      if (aChecked != bChecked) {
        return aChecked ? 1 : -1;
      }
      
      final tA = DateTime.parse(a.timestamp).toLocal();
      final tB = DateTime.parse(b.timestamp).toLocal();
      return tB.compareTo(tA);
    });

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF181924),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: const [
              Icon(Icons.mic_none, size: 18, color: Color(0xFFB5BAD3)),
              SizedBox(width: 6),
              Text(
                '오디오 이벤트 내역',
                style: TextStyle(
                  color: Color(0xFFB5BAD3),
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: audioEvents.isEmpty
                ? const Center(
                    child: Text(
                      '새로운 오디오 이벤트가 없습니다.',
                      style: TextStyle(color: Color(0xFF4A4E63)),
                    ),
                  )
                : ListView.separated(
                    itemBuilder: (context, index) {
                      final event = audioEvents[index];
                      return _AudioTile(event: event, showOnlyUnchecked: showOnlyUnchecked);
                    },
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemCount: audioEvents.length,
                  ),
          ),
        ],
      ),
    );
  }
}

class _AudioTile extends ConsumerStatefulWidget {
  final AudioEvent event;
  final bool showOnlyUnchecked;

  const _AudioTile({required this.event, this.showOnlyUnchecked = false});

  @override
  ConsumerState<_AudioTile> createState() => _AudioTileState();
}

class _AudioTileState extends ConsumerState<_AudioTile> {
  late AudioPlayer _audioPlayer;
  bool isPlaying = false;

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    _audioPlayer.onPlayerStateChanged.listen((state) {
      if (mounted) {
        setState(() {
          isPlaying = state == PlayerState.playing;
        });
      }
    });
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }

  void _togglePlay() async {
    final config = ref.read(serverConfigProvider);
    if (isPlaying) {
      await _audioPlayer.pause();
    } else {
      final url = '${config.baseUrl}${widget.event.audioUrl}';
      await _audioPlayer.play(UrlSource(url));
    }
  }

  void _showLabelDialog() {
    showDialog(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          backgroundColor: const Color(0xFF1C1E2B),
          title: const Text('오디오 라벨 지정', style: TextStyle(color: Colors.white)),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                title: const Text('비명 소리 (Scream)', style: TextStyle(color: Colors.white70)),
                onTap: () => _updateLabel('비명 소리'),
              ),
              ListTile(
                title: const Text('유리 깨짐 (Glass Break)', style: TextStyle(color: Colors.white70)),
                onTap: () => _updateLabel('유리 깨짐'),
              ),
              ListTile(
                title: const Text('폭발음 (Explosion)', style: TextStyle(color: Colors.white70)),
                onTap: () => _updateLabel('폭발음'),
              ),
              ListTile(
                title: const Text('기타 소음 (Noise)', style: TextStyle(color: Colors.white70)),
                onTap: () => _updateLabel('기타 소음'),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _updateLabel(String label) async {
    final config = ref.read(serverConfigProvider);
    Navigator.of(context).pop();
    try {
      await http.patch(
        Uri.parse('${config.baseUrl}/audio_events/${widget.event.audioEventId}/label'),
        headers: {'Content-Type': 'application/json'},
        body: '{"admin_label": "$label"}',
      );
      // Wait a bit and refresh provider to show updated label
      Future.delayed(const Duration(milliseconds: 500), () {
         ref.invalidate(audioEventListProvider);
      });
    } catch (e) {
      debugPrint('Label update failed: $e');
    }
  }

  String formatLocalTime(String ts) {
    try {
      final dt = DateTime.parse(ts).toLocal();
      final timeStr = '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}:${dt.second.toString().padLeft(2, '0')}';
      if (widget.showOnlyUnchecked) return timeStr;
      
      final dateStr = '${dt.year.toString().substring(2)}/${dt.month.toString().padLeft(2, '0')}/${dt.day.toString().padLeft(2, '0')}';
      return '$dateStr $timeStr';
    } catch (_) {
      return '--:--:--';
    }
  }

  @override
  Widget build(BuildContext context) {
    String formattedTime = formatLocalTime(widget.event.timestamp);
    final bool isChecked = widget.event.adminLabel != null;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF11131C),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isChecked ? const Color(0xFF2E3244) : const Color(0xFFBA68C8).withOpacity(0.5),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            icon: Icon(
              isPlaying ? Icons.pause_circle_filled : Icons.play_circle_fill,
              color: const Color(0xFF1F8CEB),
              size: 36,
            ),
            onPressed: _togglePlay,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        widget.event.modelLabel ?? '미분류 오디오',
                        style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600),
                      ),
                    ),
                    if (isChecked) ...[
                      const SizedBox(width: 4),
                      const Icon(Icons.check_circle, color: Color(0xFF4ADE80), size: 14),
                    ],
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  formattedTime,
                  style: const TextStyle(color: Color(0xFF757B92), fontSize: 11),
                ),
                if (widget.event.doa != null)
                  Text(
                    '발생 각도: ${widget.event.doa!.toStringAsFixed(1)}°',
                    style: const TextStyle(color: Color(0xFFB5BAD3), fontSize: 11),
                  ),
              ],
            ),
          ),
          isChecked
              ? Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xFF232433),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.check_circle, color: Color(0xFF4ADE80), size: 12),
                      const SizedBox(width: 4),
                      Text(
                        widget.event.adminLabel ?? '확인됨',
                        style: const TextStyle(
                          color: Color(0xFF4ADE80),
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                )
              : ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1F8CEB),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    minimumSize: Size.zero,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                  ),
                  onPressed: _showLabelDialog,
                  child: const Text('라벨 지정', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                ),
        ],
      ),
    );
  }
}
