import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../utils/map_transformer.dart';
import '../utils/map_transformer_loader.dart';
import '../features/dashboard/dashboard_provider.dart';

String _getYamlPath(String locationName) {
  switch (locationName) {
    case '데이터센터 3층 전산실':
      return 'assets/map_metadata/map_v1_debug.yaml';
    case '데이터센터 2층 서버룸':
      return 'assets/map_metadata/map_test_ss.yaml';
    default:
      return 'assets/map_metadata/map_v1_debug.yaml';
  }
}

final mapTransformerProvider = FutureProvider<MapTransformer>((ref) async {
  final location = ref.watch(dashboardMapLocationProvider);
  final loader = MapTransformerLoader(yamlAssetPath: _getYamlPath(location));
  return await loader.load();
});

final mapImagePathProvider = FutureProvider<String>((ref) async {
  final location = ref.watch(dashboardMapLocationProvider);
  final loader = MapTransformerLoader(yamlAssetPath: _getYamlPath(location));
  return await loader.loadMapImagePath();
});
