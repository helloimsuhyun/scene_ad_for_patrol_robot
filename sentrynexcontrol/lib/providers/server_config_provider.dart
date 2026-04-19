import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ServerConfig {
  final String serverIp;
  final String port;

  ServerConfig({required this.serverIp, this.port = '8000'});

  String get baseUrl => 'http://$serverIp:$port';
  String get imageUrlBase => 'http://$serverIp:$port/images/';
  String get audioUrlBase => 'http://$serverIp:$port/audio/';
  String get yoloImageUrlBase => 'http://$serverIp:$port/person_images/';

  ServerConfig copyWith({String? serverIp, String? port}) {
    return ServerConfig(
      serverIp: serverIp ?? this.serverIp,
      port: port ?? this.port,
    );
  }
}

class ServerConfigNotifier extends StateNotifier<ServerConfig> {
  ServerConfigNotifier() : super(ServerConfig(serverIp: '127.0.0.1')) {
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
