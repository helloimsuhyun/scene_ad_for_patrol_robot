import 'package:flutter/material.dart';
import '../../widgets/alert_list.dart';

class LogsScreen extends StatelessWidget {
  const LogsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.transparent,
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          SizedBox(height: 8),
          Text(
            '데이터베이스에 저장된 모든 비정상/정상 이벤트를 조회합니다 (최신순).',
            style: TextStyle(
              color: Color(0xFF7A7F96),
              fontSize: 14,
            ),
          ),
          SizedBox(height: 24),
          Expanded(child: AlertList()),
        ],
      ),
    );
  }
}
