import 'dart:convert';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;

// 시그널링 서버 주소 (signaling_server.py - 포트 8001)
const String _signalingUrl = 'http://127.0.0.1:8001';

class CameraStreamWidget extends StatefulWidget {
  const CameraStreamWidget({super.key});

  @override
  State<CameraStreamWidget> createState() => _CameraStreamWidgetState();
}

class _CameraStreamWidgetState extends State<CameraStreamWidget> {
  final RTCVideoRenderer _remoteRenderer = RTCVideoRenderer();
  RTCPeerConnection? _peerConnection;

  _Status _status = _Status.idle;
  bool _rendererReady = false;

  @override
  void initState() {
    super.initState();
    _initRenderer();
  }

  Future<void> _initRenderer() async {
    await _remoteRenderer.initialize();
    _remoteRenderer.srcObject = null;
    if (mounted) {
      setState(() => _rendererReady = true);
    }
  }

  @override
  void dispose() {
    _remoteRenderer.dispose();
    _peerConnection?.close();
    super.dispose();
  }

  Future<void> _waitIceGatheringComplete(RTCPeerConnection pc) async {
    if (pc.iceGatheringState ==
        RTCIceGatheringState.RTCIceGatheringStateComplete) {
      return;
    }

    final completer = Completer<void>();

    pc.onIceGatheringState = (state) {
      debugPrint('[WebRTC] iceGatheringState=$state');
      if (state == RTCIceGatheringState.RTCIceGatheringStateComplete) {
        if (!completer.isCompleted) {
          completer.complete();
        }
      }
    };

    await completer.future.timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        debugPrint('[WebRTC] ICE gathering timeout');
      },
    );
  }

  Future<void> _connect() async {
    setState(() => _status = _Status.connecting);

    try {
      if (!_rendererReady) {
        await _initRenderer();
      }

      final pc = await createPeerConnection({
        'iceServers': [
          {'urls': 'stun:stun.l.google.com:19302'},
        ],
      });

      _peerConnection = pc;

      pc.onTrack = (event) async {
        debugPrint(
          '[WebRTC] onTrack kind=${event.track.kind}, streams=${event.streams.length}',
        );

        if (event.track.kind != 'video') return;

        MediaStream? stream;

        if (event.streams.isNotEmpty) {
          stream = event.streams.first;
        } else {
          stream = await createLocalMediaStream('remote_stream');
          await stream.addTrack(event.track);
        }

        _remoteRenderer.srcObject = stream;

        Future.delayed(const Duration(seconds: 1), () {
          debugPrint(
            '[WebRTC] renderer size: '
            '${_remoteRenderer.videoWidth} x ${_remoteRenderer.videoHeight}',
          );
        });

        if (mounted) {
          setState(() => _status = _Status.connected);
        }
      };

      pc.onAddStream = (stream) {
        debugPrint('[WebRTC] onAddStream id=${stream.id}');
        _remoteRenderer.srcObject = stream;
        if (mounted) {
          setState(() => _status = _Status.connected);
        }
      };

      pc.onConnectionState = (state) {
        debugPrint('[WebRTC] connectionState=$state');

        if (state == RTCPeerConnectionState.RTCPeerConnectionStateFailed ||
            state ==
                RTCPeerConnectionState.RTCPeerConnectionStateDisconnected) {
          if (mounted) setState(() => _status = _Status.idle);
        }
      };

      pc.onIceConnectionState = (state) {
        debugPrint('[WebRTC] iceConnectionState=$state');
      };

      await pc.addTransceiver(
        kind: RTCRtpMediaType.RTCRtpMediaTypeVideo,
        init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
      );

      final offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await _waitIceGatheringComplete(pc);

      final localDesc = await pc.getLocalDescription();
      if (localDesc == null) {
        throw Exception('localDescription is null');
      }

      debugPrint(
        '[WebRTC] sending viewer_offer type=${localDesc.type}, '
        'sdp_len=${localDesc.sdp?.length ?? 0}',
      );

      final resp = await http.post(
        Uri.parse('$_signalingUrl/viewer_offer'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'sdp': localDesc.sdp,
          'type': localDesc.type,
        }),
      );

      if (resp.statusCode != 200) {
        throw Exception('Signaling failed: ${resp.statusCode} ${resp.body}');
      }

      final answerJson = jsonDecode(resp.body) as Map<String, dynamic>;
      final answer = RTCSessionDescription(
        answerJson['sdp'],
        answerJson['type'],
      );

      await pc.setRemoteDescription(answer);
    } catch (e) {
      debugPrint('[WebRTC] ERROR: $e');
      if (mounted) setState(() => _status = _Status.error);
    }
  }

  Future<void> _disconnect() async {
    await _peerConnection?.close();
    _peerConnection = null;
    _remoteRenderer.srcObject = null;
    if (mounted) setState(() => _status = _Status.idle);
  }

  @override
  Widget build(BuildContext context) {
    // 1. 연결 완료 상태 (full-bleed)
    if (_status == _Status.connected) {
      return Container(
        decoration: BoxDecoration(
          color: const Color(0xFF0D0E16),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF2D3041)),
        ),
        clipBehavior: Clip.antiAlias,
        child: AspectRatio(
          aspectRatio: 16 / 9,
          child: Stack(
            children: [
              // 비전 스트림 화면 (꽉 채우기)
              Positioned.fill(
                child: _buildVideoArea(),
              ),
              // 우측 하단 전체화면 버튼
              Positioned(
                right: 8,
                bottom: 8,
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: IconButton(
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    icon: const Icon(
                      Icons.fullscreen,
                      size: 22,
                      color: Colors.white,
                    ),
                    onPressed: () => _showFullScreen(context),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    // 2. 연결 전/대기 상태 (padded layout)
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF0D0E16),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF2D3041)),
      ),
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(
                Icons.videocam_outlined,
                size: 14,
                color: Color(0xFFB5BAD3),
              ),
              const SizedBox(width: 6),
              const Expanded(
                child: Text(
                  '로봇 카메라',
                  style: TextStyle(
                    color: Color(0xFFB5BAD3),
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              _StatusDot(status: _status),
            ],
          ),
          const SizedBox(height: 10),
          AspectRatio(
            aspectRatio: 16 / 9,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: _buildVideoArea(),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(width: double.infinity, child: _buildConnectButton()),
        ],
      ),
    );
  }

  Widget _buildVideoArea() {
    if (_status == _Status.connected) {
      // 영상이 연결된 경우: WebRTC 렌더러 (클릭 시 연결 해제 옵션 제공)
      return Stack(
        children: [
          Positioned.fill(
            child: RTCVideoView(
              _remoteRenderer,
              objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
            ),
          ),
          Positioned.fill(
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                onTap: () {
                  // 한 번 클릭하면 연결 해제 확인 창 표시
                  showDialog(
                    context: context,
                    builder: (ctx) => AlertDialog(
                      backgroundColor: const Color(0xFF1C1E2B),
                      title: const Text('카메라 연결 해제', style: TextStyle(color: Colors.white)),
                      content: const Text('로봇 카메라 연결을 종료하시겠습니까?', style: TextStyle(color: Colors.white70)),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(ctx),
                          child: const Text('취소', style: TextStyle(color: Colors.white54)),
                        ),
                        ElevatedButton(
                          style: ElevatedButton.styleFrom(backgroundColor: Colors.redAccent),
                          onPressed: () {
                            Navigator.pop(ctx);
                            _disconnect();
                          },
                          child: const Text('연결 해제', style: TextStyle(color: Colors.white)),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      );
    }


    return Container(
      color: const Color(0xFF0D0E16),
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              _status == _Status.error
                  ? Icons.signal_wifi_statusbar_connected_no_internet_4_outlined
                  : Icons.videocam_off_outlined,
              color: const Color(0xFF4A4E63),
              size: 28,
            ),
            const SizedBox(height: 8),
            Text(
              _statusLabel(),
              style: const TextStyle(color: Color(0xFF4A4E63), fontSize: 11),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConnectButton() {
    final bool isIdle = _status == _Status.idle || _status == _Status.error;

    return TextButton(
      style: TextButton.styleFrom(
        backgroundColor: isIdle
            ? const Color(0xFF1F8CEB).withValues(alpha: 0.15)
            : const Color(0xFFEF4444).withValues(alpha: 0.12),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(vertical: 8),
      ),
      onPressed: _status == _Status.connecting
          ? null
          : (isIdle ? _connect : _disconnect),
      child: _status == _Status.connecting
          ? const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.white54,
              ),
            )
          : Text(
              isIdle ? '연결하기' : '끊기',
              style: TextStyle(
                fontSize: 12,
                color: isIdle
                    ? const Color(0xFF1F8CEB)
                    : const Color(0xFFEF4444),
              ),
            ),
    );
  }

  String _statusLabel() {
    switch (_status) {
      case _Status.idle:
        return '아래 버튼을 눌러\n카메라에 연결하세요.';
      case _Status.connecting:
        return '연결 중...';
      case _Status.connected:
        return '스트리밍 중';
      case _Status.error:
        return '연결 실패\n로봇 송신기 확인 후 재시도';
    }
  }

  String _statusLabel2() {
    switch (_status) {
      case _Status.idle:
        return '';
      case _Status.connecting:
        return '';
      case _Status.connected:
        return '실시간 로봇 카메라 송신 중';
      case _Status.error:
        return '';
    }
  }

  void _showFullScreen(BuildContext context) {
    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Close',
      barrierColor: Colors.black.withValues(alpha: 0.85),
      transitionDuration: const Duration(milliseconds: 200),
      pageBuilder: (context, anim1, anim2) {
        return Center(
          child: Container(
            width: MediaQuery.of(context).size.width * 0.8,
            constraints: const BoxConstraints(maxWidth: 1200),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Align(
                  alignment: Alignment.topRight,
                  child: IconButton(
                    icon: const Icon(
                      Icons.close,
                      color: Colors.white,
                      size: 30,
                    ),
                    onPressed: () => Navigator.pop(context),
                  ),
                ),
                AspectRatio(
                  aspectRatio: 16 / 9,
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.black,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0xFF2D3041)),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: _buildVideoArea(),
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  _statusLabel2(),
                  style: const TextStyle(
                    color: Color(0xFF7F7CFF),
                    fontSize: 16,
                    decoration: TextDecoration.none,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

enum _Status { idle, connecting, connected, error }

class _StatusDot extends StatelessWidget {
  final _Status status;
  const _StatusDot({required this.status, super.key});

  @override
  Widget build(BuildContext context) {
    Color color;
    switch (status) {
      case _Status.connected:
        color = const Color(0xFF22C55E);
        break;
      case _Status.connecting:
        color = const Color(0xFFEAB308);
        break;
      case _Status.error:
        color = const Color(0xFFEF4444);
        break;
      case _Status.idle:
        color = const Color(0xFF4A4E63);
        break;
    }
    return Container(
      width: 8,
      height: 8,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
    );
  }
}