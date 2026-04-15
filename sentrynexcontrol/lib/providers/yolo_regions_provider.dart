import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'server_config_provider.dart';

class YoloRegionsNotifier extends StateNotifier<AsyncValue<List<Map<String, dynamic>>>> {
  final Ref ref;
  YoloRegionsNotifier(this.ref) : super(const AsyncValue.loading()) {
    fetchRegions();
  }

  Future<void> fetchRegions() async {
    final config = ref.read(serverConfigProvider);
    state = const AsyncValue.loading();
    try {
      final res = await http.get(Uri.parse('${config.baseUrl}/robot/yolo_regions'));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        if (data['ok'] == true) {
          state = AsyncValue.data(List<Map<String, dynamic>>.from(data['regions']));
          return;
        }
      }
      state = AsyncValue.error('Fetch failed', StackTrace.current);
    } catch (e, st) {
      state = AsyncValue.error(e, st);
    }
  }

  Future<void> addRegion(String name, double xMin, double xMax, double yMin, double yMax) async {
    final config = ref.read(serverConfigProvider);
    try {
      await http.post(
        Uri.parse('${config.baseUrl}/robot/yolo_regions'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'name': name,
          'x_min': xMin,
          'x_max': xMax,
          'y_min': yMin,
          'y_max': yMax,
          'is_enabled': true,
        }),
      );
      await fetchRegions();
    } catch (e) {
      // ignore
    }
  }

  Future<void> deleteRegion(int regionId) async {
    final config = ref.read(serverConfigProvider);
    try {
      await http.delete(Uri.parse('${config.baseUrl}/robot/yolo_regions/$regionId'));
      await fetchRegions();
    } catch (e) {
      // ignore
    }
  }

  Future<void> toggleRegionEnabled(int regionId, bool isEnabled) async {
    final config = ref.read(serverConfigProvider);
    try {
      await http.patch(
        Uri.parse('${config.baseUrl}/robot/yolo_regions/$regionId/enabled'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'is_enabled': isEnabled}),
      );
      await fetchRegions();
    } catch (e) {
      // ignore
    }
  }
}

final yoloRegionsProvider = StateNotifierProvider<YoloRegionsNotifier, AsyncValue<List<Map<String, dynamic>>>>((ref) {
  return YoloRegionsNotifier(ref);
});
