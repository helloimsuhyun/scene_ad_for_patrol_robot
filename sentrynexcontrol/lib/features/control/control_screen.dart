import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'control_provider.dart';

class ControlScreen extends ConsumerWidget {
  const ControlScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final placesAsync = ref.watch(placesProvider);

    return Container(
      color: const Color(0xFF0A0B10), // 배경색 통일
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          //---------- 상단 타이틀 및 마스터 버튼 영역 ----------
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '시스템 제어 센터',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Row(
                children: [
                  ElevatedButton.icon(
                    onPressed: () => ControlActions.recalibrateAll(ref),
                    icon: const Icon(Icons.sync, size: 18),
                    label: const Text(
                      '전체 재학습',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1F8CEB),
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 16,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton.icon(
                    onPressed: () => ControlActions.deleteAllPlaces(ref),
                    icon: const Icon(
                      Icons.delete_forever,
                      size: 18,
                      color: Color(0xFFFF4B5C),
                    ),
                    label: const Text(
                      '전체 구역 삭제',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Color(0xFFFF4B5C),
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(
                        color: Color(0xFFFF4B5C),
                        width: 1.5,
                      ),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 16,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),

          //---------- 데이터 리스트 영역 ----------
          Expanded(
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                color: const Color(0xFF181924),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF2D3041)),
              ),
              child: placesAsync.when(
                data: (data) {
                  final places = data['places'] as List<dynamic>? ?? [];
                  if (places.isEmpty) {
                    return const Center(
                      child: Text(
                        '등록된 구역이 없습니다.',
                        style: TextStyle(color: Colors.white70),
                      ),
                    );
                  }

                  return LayoutBuilder(
                    builder: (context, constraints) {
                      return SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: ConstrainedBox(
                          constraints: BoxConstraints(
                            minWidth: constraints.maxWidth,
                          ),
                          child: SingleChildScrollView(
                            child: DataTable(
                              headingRowColor: WidgetStateProperty.resolveWith(
                                (states) => const Color(0xFF26293A),
                              ),
                              dataRowColor: WidgetStateProperty.resolveWith(
                                (states) => const Color(0xFF181924),
                              ),
                              dividerThickness: 0.5,
                              horizontalMargin: 24,
                              columnSpacing: 40,
                              headingTextStyle: const TextStyle(
                                color: Color(0xFFB5BAD3),
                                fontWeight: FontWeight.bold,
                                fontSize: 14,
                              ),
                              dataTextStyle: const TextStyle(
                                color: Colors.white70,
                                fontSize: 13,
                              ),
                              columns: const [
                                DataColumn(label: Text('구역 (Place)')),
                                DataColumn(label: Text('모드 (Mode)')),
                                DataColumn(label: Text('뱅크 (Bank)')),
                                DataColumn(label: Text('임계 카운트')),
                                DataColumn(label: Text('재학습 필요')),
                                DataColumn(label: Text('관리 기능')),
                              ],
                              rows: places.map((p) {
                                final placeId = p['place_id'].toString();
                                final mode = p['mode'].toString();
                                final bankCount = p['bank_count'];
                                final bankTarget = p['bank_target'];
                                final thCalibCount = p['th_calib_count'];
                                final thCalibTarget = p['th_calib_target'];
                                final needCalibration =
                                    p['need_calibration'] == true;

                                return DataRow(
                                  cells: [
                                    DataCell(
                                      Text(
                                        placeId,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.bold,
                                          color: Colors.white,
                                        ),
                                      ),
                                    ),
                                    DataCell(
                                      DropdownButton<String>(
                                        value: mode,
                                        dropdownColor: const Color(0xFF26293A),
                                        style: const TextStyle(
                                          color: Colors.white,
                                        ),
                                        underline: const SizedBox(),
                                        items: const [
                                          DropdownMenuItem(
                                            value: 'idle',
                                            child: Text('idle'),
                                          ),
                                          DropdownMenuItem(
                                            value: 'bank',
                                            child: Text('bank'),
                                          ),
                                          DropdownMenuItem(
                                            value: 'th_calib',
                                            child: Text('th_calib'),
                                          ),
                                          DropdownMenuItem(
                                            value: 'query',
                                            child: Text('query'),
                                          ),
                                        ],
                                        onChanged: (newMode) {
                                          if (newMode != null &&
                                              newMode != mode) {
                                            ControlActions.setMode(
                                              ref,
                                              placeId,
                                              newMode,
                                            );
                                          }
                                        },
                                      ),
                                    ),
                                    DataCell(Text('$bankCount / $bankTarget')),
                                    DataCell(
                                      Text('$thCalibCount / $thCalibTarget'),
                                    ),
                                    DataCell(
                                      Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 8,
                                          vertical: 4,
                                        ),
                                        decoration: BoxDecoration(
                                          color: needCalibration
                                              ? Colors.red.withOpacity(0.2)
                                              : Colors.green.withOpacity(0.2),
                                          borderRadius: BorderRadius.circular(
                                            4,
                                          ),
                                        ),
                                        child: Text(
                                          needCalibration ? '필요 (1)' : '완료 (0)',
                                          style: TextStyle(
                                            color: needCalibration
                                                ? Colors.redAccent
                                                : Colors.greenAccent,
                                            fontSize: 12,
                                          ),
                                        ),
                                      ),
                                    ),
                                    DataCell(
                                      Row(
                                        children: [
                                          TextButton(
                                            onPressed: () =>
                                                ControlActions.deleteThreshold(
                                                  ref,
                                                  placeId,
                                                ),
                                            child: const Text(
                                              '임계치 삭제',
                                              style: TextStyle(
                                                color: Colors.yellowAccent,
                                              ),
                                            ),
                                          ),
                                          TextButton(
                                            onPressed: () =>
                                                ControlActions.deletePlace(
                                                  ref,
                                                  placeId,
                                                ),
                                            child: const Text(
                                              '구역 삭제',
                                              style: TextStyle(
                                                color: Colors.redAccent,
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                );
                              }).toList(),
                            ),
                          ),
                        ),
                      );
                    },
                  );
                },
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (err, stack) => Center(
                  child: Text(
                    '에러 발생: $err',
                    style: const TextStyle(color: Colors.red),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
