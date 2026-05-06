import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:yaml/yaml.dart';

import 'map_transformer.dart';

class MapTransformerLoader {
  final String yamlAssetPath;

  const MapTransformerLoader({
    required this.yamlAssetPath,
  });

  Future<MapTransformer> load() async {
    final yamlStr = await rootBundle.loadString(yamlAssetPath);
    final yamlMap = loadYaml(yamlStr);

    final resolution = (yamlMap['resolution'] as num).toDouble();

    final origin = yamlMap['origin'] as YamlList;
    final originX = (origin[0] as num).toDouble();
    final originY = (origin[1] as num).toDouble();

    final imageName = yamlMap['image'] as String;
    final yamlDir = yamlAssetPath.substring(0, yamlAssetPath.lastIndexOf('/'));
    final imageAssetPath = '$yamlDir/$imageName';

    final imageBytes = await rootBundle.load(imageAssetPath);
    final codec = await ui.instantiateImageCodec(imageBytes.buffer.asUint8List());
    final frame = await codec.getNextFrame();
    final image = frame.image;

    return MapTransformer(
      resolution: resolution,
      originX: originX,
      originY: originY,
      imageHeight: image.height.toDouble(),
      imageWidth: image.width.toDouble(),
    );
  }

  Future<String> loadMapImagePath() async {
    final yamlStr = await rootBundle.loadString(yamlAssetPath);
    final yamlMap = loadYaml(yamlStr);

    final imageName = yamlMap['image'] as String;
    final yamlDir = yamlAssetPath.substring(0, yamlAssetPath.lastIndexOf('/'));
    return '$yamlDir/$imageName';
  }
}
