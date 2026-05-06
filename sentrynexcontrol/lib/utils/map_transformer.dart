class MapTransformer {
  final double resolution;
  final double originX;
  final double originY;
  final double imageHeight;
  final double imageWidth;

  MapTransformer({
    required this.resolution,
    required this.originX,
    required this.originY,
    required this.imageHeight,
    required this.imageWidth,
  });

  Map<String, double> transform(double x, double y, double yaw) {
    final px = (x - originX) / resolution;
    final py = imageHeight - ((y - originY) / resolution);
    final yawGui = -yaw;

    return {
      "px": px,
      "py": py,
      "yaw": yawGui,
    };
  }

  Map<String, double> inverseTransform(double px, double py, double yawGui) {
    final x = (px * resolution) + originX;
    final y = originY + ((imageHeight - py) * resolution);
    final yaw = -yawGui;

    return {
      "x": x,
      "y": y,
      "yaw": yaw,
    };
  }
}
