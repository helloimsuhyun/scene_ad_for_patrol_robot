import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ServerConfig {
  final String serverIp;
  final String port;

  ServerConfig({required this.serverIp, this.port = '8000'});

  // ngrok 주소면 https를 쓰고 포트 생략, 아니면 기존 http://$ip:$port 사용
  bool get _isExternal => serverIp.contains('ngrok-free.app') || serverIp.startsWith('http');

  String get _formattedBase {
    if (serverIp.startsWith('http')) return serverIp;
    if (serverIp.contains('ngrok-free.app')) return 'https://$serverIp';
    return 'http://$serverIp:$port';
  }

  String get baseUrl => _formattedBase;
  String get imageUrlBase => '$_formattedBase/images/';
  String get audioUrlBase => '$_formattedBase/audio/';
  String get yoloImageUrlBase => '$_formattedBase/person_images/';

  ServerConfig copyWith({String? serverIp, String? port}) {
    return ServerConfig(
      serverIp: serverIp ?? this.serverIp,
      port: port ?? this.port,
    );
  }
}

class ServerConfigNotifier extends StateNotifier<ServerConfig> {
  ServerConfigNotifier() : super(ServerConfig(serverIp: 'hungrily-pasted-pursuant.ngrok-free.app')) {
    _loadConfig();
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
