/*
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
*/


import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'server_config_provider.dart';

class YoloRegionsNotifier
    extends StateNotifier<AsyncValue<List<Map<String, dynamic>>>> {
  final Ref ref;

  YoloRegionsNotifier(this.ref) : super(const AsyncValue.loading()) {
    fetchRegions();
  }

  List<Map<String, dynamic>> get _currentRegions {
    return state.value ?? <Map<String, dynamic>>[];
  }

  int? _toInt(dynamic value) {
    if (value is int) return value;
    return int.tryParse(value.toString());
  }

  Future<void> fetchRegions() async {
    final config = ref.read(serverConfigProvider);

    // 기존 데이터가 없을 때만 loading 표시
    // 기존 데이터가 있으면 지도에서 기존 구역을 유지
    if (state.value == null) {
      state = const AsyncValue.loading();
    }

    try {
      final res = await http.get(
        Uri.parse('${config.baseUrl}/robot/yolo_regions'),
      );

      if (res.statusCode != 200) {
        throw Exception('Fetch regions failed: ${res.statusCode}, ${res.body}');
      }

      final data = jsonDecode(res.body);

      if (data['ok'] != true) {
        throw Exception('Fetch regions failed: ok=false, ${res.body}');
      }

      state = AsyncValue.data(
        List<Map<String, dynamic>>.from(data['regions']),
      );
    } catch (e, st) {
      if (state.value != null) {
        state = AsyncValue.data(_currentRegions);
      } else {
        state = AsyncValue.error(e, st);
      }
    }
  }

  Future<void> addRegion(
    String name,
    double xMin,
    double xMax,
    double yMin,
    double yMax,
  ) async {
    final config = ref.read(serverConfigProvider);

    final previous = List<Map<String, dynamic>>.from(_currentRegions);

    // 서버 DB에 저장되기 전 임시 ID
    final tempId = -DateTime.now().millisecondsSinceEpoch;

    final tempRegion = <String, dynamic>{
      'region_id': tempId,
      'name': name,
      'x_min': xMin,
      'x_max': xMax,
      'y_min': yMin,
      'y_max': yMax,
      'is_enabled': true,
      'is_pending': true,
    };

    // 1) 지도에 먼저 임시 구역 표시
    state = AsyncValue.data([...previous, tempRegion]);

    try {
      final res = await http.post(
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

      if (res.statusCode != 200) {
        throw Exception('Add region failed: ${res.statusCode}, ${res.body}');
      }

      // 2) 서버 DB 기준 실제 region_id로 다시 동기화
      await fetchRegions();
    } catch (e) {
      // 실패하면 임시 구역 제거
      state = AsyncValue.data(previous);
    }
  }

  Future<int?> _resolveRegionId(int regionId) async {
    // 실제 DB에 저장된 region_id만 처리
    if (regionId >= 0) return regionId;

    // 임시 구역(region_id < 0)은 저장 중 상태이므로 입력 무시
    return null;
  }

  Future<void> deleteRegion(int regionId) async {
    final actualRegionId = await _resolveRegionId(regionId);
    if (actualRegionId == null) return;

    final config = ref.read(serverConfigProvider);

    final previous = List<Map<String, dynamic>>.from(_currentRegions);

    // 1) 화면에서 먼저 제거
    state = AsyncValue.data(
      previous.where((r) => _toInt(r['region_id']) != actualRegionId).toList(),
    );

    try {
      final res = await http.delete(
        Uri.parse('${config.baseUrl}/robot/yolo_regions/$actualRegionId'),
      );

      if (res.statusCode != 200) {
        throw Exception('Delete region failed: ${res.statusCode}, ${res.body}');
      }

      // 2) 서버 DB 기준 재동기화
      await fetchRegions();
    } catch (e) {
      // 실패하면 복구
      state = AsyncValue.data(previous);
    }
  }

  Future<void> toggleRegionEnabled(int regionId, bool isEnabled) async {
    final actualRegionId = await _resolveRegionId(regionId);
    if (actualRegionId == null) return;

    final config = ref.read(serverConfigProvider);

    final previous = List<Map<String, dynamic>>.from(_currentRegions);

    // 1) 화면에서 먼저 ON/OFF 반영
    final updated = previous.map((r) {
      if (_toInt(r['region_id']) == actualRegionId) {
        return {
          ...r,
          'is_enabled': isEnabled,
        };
      }
      return r;
    }).toList();

    state = AsyncValue.data(updated);

    try {
      final res = await http.patch(
        Uri.parse('${config.baseUrl}/robot/yolo_regions/$actualRegionId/enabled'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'is_enabled': isEnabled}),
      );

      if (res.statusCode != 200) {
        throw Exception('Toggle region failed: ${res.statusCode}, ${res.body}');
      }

      // 2) 서버 DB 기준 재동기화
      await fetchRegions();
    } catch (e) {
      // 실패하면 복구
      state = AsyncValue.data(previous);
    }
  }
}

final yoloRegionsProvider = StateNotifierProvider<
    YoloRegionsNotifier,
    AsyncValue<List<Map<String, dynamic>>>>((ref) {
  return YoloRegionsNotifier(ref);
});