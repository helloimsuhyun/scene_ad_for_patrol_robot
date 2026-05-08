import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

class ServerConfig {
  final String serverIp;
  final String port;

  ServerConfig({required this.serverIp, this.port = '8000'});

  String get baseUrl => 'http://$serverIp:$port';
  String get signalingUrl => 'http://$serverIp:8001'; // 시그널링 서버 (고정 포트 8001)
  String get imageUrlBase => 'http://$serverIp:$port/images/';
  String get audioUrlBase => 'http://$serverIp:$port/audio/';
  String get yoloImageUrlBase => 'http://$serverIp:$port/person_images/';
  String get authImageUrlBase => 'http://$serverIp:$port/auth_images/';

  String getUrl(String path) {
    if (path.startsWith('/')) {
      return '$baseUrl$path';
    }
    return '$baseUrl/$path';
  }

  ServerConfig copyWith({String? serverIp, String? port}) {
    return ServerConfig(
      serverIp: serverIp ?? this.serverIp,
      port: port ?? this.port,
    );
  }
}

class ServerConfigNotifier extends StateNotifier<ServerConfig> {
  ServerConfigNotifier() : super(ServerConfig(serverIp: _getDefaultIp())) {
    _loadConfig();
  }

  static String _getDefaultIp() {
    if (kIsWeb) {
      final host = Uri.base.host;
      if (host.isNotEmpty) {
        return host;
      }
    }
    return '127.0.0.1';
  }

  static const String _keyIp = 'server_ip';

  Future<void> _loadConfig() async {
    final prefs = await SharedPreferences.getInstance();
    final savedIp = prefs.getString(_keyIp);
    if (savedIp != null && savedIp.isNotEmpty) {
      state = state.copyWith(serverIp: savedIp);
    }
  }

  Future<void> setIp(String ip) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyIp, ip);
    state = state.copyWith(serverIp: ip);
  }
}

final serverConfigProvider = StateNotifierProvider<ServerConfigNotifier, ServerConfig>((ref) {
  return ServerConfigNotifier();
});
