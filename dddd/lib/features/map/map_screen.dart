import 'package:flutter/material.dart';

class MapScreen extends StatelessWidget {
  const MapScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF11121A),
      padding: const EdgeInsets.all(24),
      child: const Center(
        child: Text(
          '분석 / 지도 화면 (추후 구현 예정)',
          style: TextStyle(color: Colors.white70, fontSize: 16),
        ),
      ),
    );
  }
}
