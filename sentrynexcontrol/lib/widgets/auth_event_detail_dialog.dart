import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/auth_event_model.dart';
import '../providers/auth_event_provider.dart';
import '../providers/server_config_provider.dart';

class AuthEventDetailDialog extends ConsumerWidget {
  final AuthEvent event;

  const AuthEventDetailDialog({super.key, required this.event});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final serverConfig = ref.watch(serverConfigProvider);
    final baseUrl = serverConfig.baseUrl;

    Color statusColor;
    String statusText;

    switch (event.status) {
      case 'waiting_rfid':
        statusColor = const Color(0xFFFACC15);
        statusText = '인증 진행중';
        break;
      case 'success':
        statusColor = const Color(0xFF4ADE80);
        statusText = '인증 성공';
        break;
      case 'fail':
        statusColor = const Color(0xFFEF4444);
        statusText = '인증 실패';
        break;
      case 'timeout':
      default:
        statusColor = const Color(0xFF9FA4B9);
        statusText = '시간 초과 / 미인증';
        break;
    }

    return Dialog(
      backgroundColor: const Color(0xFF1C1E2B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 620),
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
            // 헤더
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.security, color: Color(0xFFFACC15), size: 24),
                    const SizedBox(width: 8),
                    const Text(
                      '2차 인증 현장 상세',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.white54),
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ],
            ),
            const Divider(color: Color(0xFF2D3041), height: 32),

            // 내용 영역
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 이미지 (있을 경우)
                Expanded(
                  flex: 3,
                  child: Container(
                    height: 250,
                    decoration: BoxDecoration(
                      color: const Color(0xFF0A0B10),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: const Color(0xFF2D3041)),
                    ),
                    child: event.imageUrl != null && event.imageUrl!.isNotEmpty
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.network(
                              '$baseUrl${event.imageUrl}',
                              fit: BoxFit.cover,
                              errorBuilder: (context, error, stackTrace) => const Center(
                                child: Icon(Icons.broken_image, color: Colors.white24, size: 48),
                              ),
                            ),
                          )
                        : const Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.camera_alt_outlined, color: Colors.white24, size: 48),
                                SizedBox(height: 8),
                                Text('이미지 대기 중...', style: TextStyle(color: Colors.white54)),
                              ],
                            ),
                          ),
                  ),
                ),
                const SizedBox(width: 20),
                
                // 정보 패널
                Expanded(
                  flex: 2,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildInfoRow('상태', statusText, valueColor: statusColor),
                      const SizedBox(height: 12),
                      _buildInfoRow('발생 시간', event.timestamp.split('T').last.split('.').first),
                      const SizedBox(height: 12),
                      _buildInfoRow('구역', event.sourceRegionName ?? '알 수 없음'),
                      const SizedBox(height: 12),
                      _buildInfoRow('결과', event.resultMessage ?? '-'),
                      
                      if (event.employeeName != null) ...[
                        const SizedBox(height: 12),
                        _buildInfoRow('사원', event.employeeName!),
                      ],
                      if (event.rfidUid != null) ...[
                        const SizedBox(height: 12),
                        _buildInfoRow('RFID', event.rfidUid!),
                      ],
                    ],
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 32),
            const Text(
              '관리자 판독',
              style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
            ),
            const SizedBox(height: 12),
            
            // 관리자 판독 액션 버튼
            Row(
              children: [
                Expanded(
                  child: _buildAdminButton(
                    context, 
                    ref, 
                    label: '오탐지 (False)', 
                    value: 'false_alarm', 
                    color: const Color(0xFFEF4444)
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildAdminButton(
                    context, 
                    ref, 
                    label: '실제 침입 (Intrusion)', 
                    value: 'real_intrusion', 
                    color: const Color(0xFFFACC15)
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildAdminButton(
                    context, 
                    ref, 
                    label: '인가자 (Authorized)', 
                    value: 'authorized', 
                    color: const Color(0xFF4ADE80)
                  ),
                ),
              ],
            ),
            
            if (event.adminLabel != null) ...[
              const SizedBox(height: 16),
              Center(
                child: Text(
                  '현재 처리 상태: ${event.adminLabel}',
                  style: const TextStyle(color: Color(0xFF9FA4B9), fontSize: 12),
                ),
              ),
            ]
          ],
          ),        // Column
        ),          // SingleChildScrollView
        ),          // ConstrainedBox
    );
  }

  Widget _buildInfoRow(String label, String value, {Color? valueColor}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(color: Color(0xFF7A7F96), fontSize: 12),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: valueColor ?? Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildAdminButton(BuildContext context, WidgetRef ref, {required String label, required String value, required Color color}) {
    final bool isSelected = event.adminLabel == value;
    
    return OutlinedButton(
      style: OutlinedButton.styleFrom(
        foregroundColor: isSelected ? Colors.black : color,
        backgroundColor: isSelected ? color : Colors.transparent,
        side: BorderSide(color: color),
        padding: const EdgeInsets.symmetric(vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
      onPressed: () {
        ref.read(authEventListProvider.notifier).updateAdminLabel(event.authEventId, value);
        Navigator.of(context).pop();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('판독이 저장되었습니다: $label')),
        );
      },
      child: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
      ),
    );
  }
}
