import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;

// 시그널링 서버 주소 (signaling_server.py - 포트 8001)
const String _signalingUrl = 'http://172.17.78.222:8001';

class CameraStreamWidget extends StatefulWidget {
  const CameraStreamWidget({super.key});

  @override
  State<CameraStreamWidget> createState() => _CameraStreamWidgetState();
}

class _CameraStreamWidgetState extends State<CameraStreamWidget> {
  // WebRTC 렌더러 (영상을 화면에 그려주는 역할)
  final RTCVideoRenderer _remoteRenderer = RTCVideoRenderer();

  RTCPeerConnection? _peerConnection;

  // 연결 상태: idle / connecting / connected / error
  _Status _status = _Status.idle;

  @override
  void initState() {
    super.initState();
    // 렌더러를 초기화해야만 영상이 제대로 표시됨
    _remoteRenderer.initialize();
  }

  @override
  void dispose() {
    _remoteRenderer.dispose();
    _peerConnection?.close();
    super.dispose();
  }

  // ─── WebRTC 연결 시작 ───
  Future<void> _connect() async {
    setState(() => _status = _Status.connecting);

    try {
      // STUN 서버 설정 (로컬이라도 ICE 후보 수집에 필요)
      final pc = await createPeerConnection({
        'iceServers': [
          {'urls': 'stun:stun.l.google.com:19302'},
        ]
      });
      _peerConnection = pc;

      // 원격(로봇)에서 보내오는 영상 트랙을 렌더러에 연결
      pc.onTrack = (event) {
        if (event.streams.isNotEmpty) {
          _remoteRenderer.srcObject = event.streams.first;
          if (mounted) setState(() => _status = _Status.connected);
        }
      };

      // 연결 끊어지면 상태를 idle로 복귀
      pc.onConnectionState = (state) {
        if (state == RTCPeerConnectionState.RTCPeerConnectionStateFailed ||
            state == RTCPeerConnectionState.RTCPeerConnectionStateDisconnected) {
          if (mounted) setState(() => _status = _Status.idle);
        }
      };

      // 영상 수신 전용 트랜시버 추가 (sendrecv 대신 recvonly)
      await pc.addTransceiver(
        kind: RTCRtpMediaType.RTCRtpMediaTypeVideo,
        init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
      );

      // SDP Offer 생성
      final offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // 시그널링 서버에 Offer 전송 → Answer 수신
      final resp = await http.post(
        Uri.parse('$_signalingUrl/viewer_offer'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'sdp': offer.sdp, 'type': offer.type}),
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
      debugPrint('[WebRTC] Error: $e');
      if (mounted) setState(() => _status = _Status.error);
    }
  }

  // ─── 연결 종료 ───
  Future<void> _disconnect() async {
    await _peerConnection?.close();
    _peerConnection = null;
    _remoteRenderer.srcObject = null;
    if (mounted) setState(() => _status = _Status.idle);
  }

  @override
  Widget build(BuildContext context) {
    //---------- 사이드바 하단 카메라 스트림 영역 ----------
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
          //---------- 카메라 헤더 ----------
          Row(
            children: [
              const Icon(Icons.videocam_outlined, size: 14, color: Color(0xFFB5BAD3)),
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
              // 전체화면 버튼
              IconButton(
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                icon: const Icon(Icons.fullscreen, size: 20, color: Color(0xFFB5BAD3)),
                onPressed: () => _showFullScreen(context),
              ),
              const SizedBox(width: 8),
              // 연결 상태 표시 점
              _StatusDot(status: _status),
            ],
          ),
          const SizedBox(height: 10),

          //---------- 영상 뷰어 영역 ----------
          AspectRatio(
            aspectRatio: 16 / 9,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: _buildVideoArea(),
            ),
          ),

          const SizedBox(height: 10),

          //---------- 연결 / 끊기 버튼 ----------
          SizedBox(
            width: double.infinity,
            child: _buildConnectButton(),
          ),
        ],
      ),
    );
  }

  // ─── 영상 뷰 또는 상태 안내 화면 표시 ───
  Widget _buildVideoArea() {
    if (_status == _Status.connected) {
      // 영상이 연결된 경우: WebRTC 렌더러
      return RTCVideoView(
        _remoteRenderer,
        objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
      );
    }

    // 영상이 없는 경우: 안내 화면
    return Container(
      color: const Color(0xFF181924),
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

  // ─── 연결 / 끊기 버튼 ───
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
          ? null // 연결 중에는 버튼 비활성화
          : (isIdle ? _connect : _disconnect),
      child: _status == _Status.connecting
          ? const SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white54),
            )
          : Text(
              isIdle ? '연결하기' : '끊기',
              style: TextStyle(
                fontSize: 12,
                color: isIdle ? const Color(0xFF1F8CEB) : const Color(0xFFEF4444),
              ),
            ),
    );
  }

  // ─── 상태별 안내 문구 ───
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

  // ─── 전체화면 확대 보기 ───
  void _showFullScreen(BuildContext context) {
    showGeneralDialog(
      context: context,
      barrierDismissible: true,
      barrierLabel: 'Close',
      barrierColor: Colors.black.withValues(alpha: 0.85), // 주변 어둡게 처리
      transitionDuration: const Duration(milliseconds: 200),
      pageBuilder: (context, anim1, anim2) {
        return Center(
          child: Container(
            width: MediaQuery.of(context).size.width * 0.8,
            constraints: const BoxConstraints(maxWidth: 1200),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // 닫기 버튼
                Align(
                  alignment: Alignment.topRight,
                  child: IconButton(
                    icon: const Icon(Icons.close, color: Colors.white, size: 30),
                    onPressed: () => Navigator.pop(context),
                  ),
                ),
                // 영상 영역
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
                const Text(
                  '실시간 로봇 카메라 송신 중',
                  style: TextStyle(color: Colors.white70, fontSize: 16, decoration: TextDecoration.none),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ─── 연결 상태 enum ───
enum _Status { idle, connecting, connected, error }

// ─── 우측 상단 상태 표시 컬러 점 ───
class _StatusDot extends StatelessWidget {
  final _Status status;
  const _StatusDot({required this.status, super.key});

  @override
  Widget build(BuildContext context) {
    Color color;
    switch (status) {
      case _Status.connected:
        color = const Color(0xFF22C55E); // 초록
        break;
      case _Status.connecting:
        color = const Color(0xFFEAB308); // 노랑
        break;
      case _Status.error:
        color = const Color(0xFFEF4444); // 빨강
        break;
      case _Status.idle:
        color = const Color(0xFF4A4E63); // 회색
        break;
    }
    return Container(
      width: 8,
      height: 8,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle),
    );
  }
}
