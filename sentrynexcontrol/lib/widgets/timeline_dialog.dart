import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../features/control/control_provider.dart';

class TimelineDialog extends ConsumerStatefulWidget {
  const TimelineDialog({super.key});

  @override
  ConsumerState<TimelineDialog> createState() => _TimelineDialogState();
}

class _TimelineDialogState extends ConsumerState<TimelineDialog> {
  int? _selectedPresetId;
  TimeOfDay? _selectedTime;

  @override
  Widget build(BuildContext context) {
    final schedulesAsync = ref.watch(schedulesProvider);
    final presetsAsync = ref.watch(presetsProvider);

    return Dialog(
      backgroundColor: const Color(0xFF161822),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        width: 600,
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '자동 순찰 타임라인 설정',
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                ),
                IconButton(
                  icon: const Icon(Icons.close, color: Colors.white54),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text('예약된 순찰 일정', style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 13)),
            const SizedBox(height: 8),
            Container(
              height: 250,
              decoration: BoxDecoration(
                color: const Color(0xFF10121A),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFF2D3041)),
              ),
              child: schedulesAsync.when(
                data: (schedules) {
                  if (schedules.isEmpty) {
                    return const Center(child: Text('설정된 예약 순찰이 없습니다.', style: TextStyle(color: Colors.white54)));
                  }
                  
                  // Sort by time
                  final sorted = List.from(schedules);
                  sorted.sort((a, b) => (a['time_str'] as String).compareTo(b['time_str'] as String));
                  
                  return presetsAsync.when(
                    data: (presets) {
                      final presetMap = {for (var p in presets) p['id']: p['name']};
                      return ListView.separated(
                        itemCount: sorted.length,
                        separatorBuilder: (_, __) => const Divider(color: Color(0xFF2D3041), height: 1),
                        itemBuilder: (context, index) {
                          final sched = sorted[index];
                          final timeStr = sched['time_str'];
                          final presetName = presetMap[sched['preset_id']] ?? '삭제된 프리셋';
                          final isActive = sched['is_active'] == 1;

                          return ListTile(
                            leading: const Icon(Icons.schedule, color: Color(0xFF7F7CFF)),
                            title: Text(timeStr, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w600)),
                            subtitle: Text('경로: $presetName', style: const TextStyle(color: Color(0xFF9FA4B9))),
                            trailing: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Switch(
                                  value: isActive,
                                  activeColor: const Color(0xFF7F7CFF),
                                  onChanged: (val) {
                                    ControlActions.toggleSchedule(ref, sched['id']);
                                  },
                                ),
                                IconButton(
                                  icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                                  onPressed: () {
                                    ControlActions.deleteSchedule(ref, sched['id']);
                                  },
                                ),
                              ],
                            ),
                          );
                        },
                      );
                    },
                    loading: () => const Center(child: CircularProgressIndicator()),
                    error: (_, __) => const Center(child: Text('오류 발생', style: TextStyle(color: Colors.red))),
                  );
                },
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (_, __) => const Center(child: Text('오류 발생', style: TextStyle(color: Colors.red))),
              ),
            ),
            const SizedBox(height: 24),
            const Text('새 일정 추가', style: TextStyle(color: Color(0xFF9FA4B9), fontSize: 13)),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: InkWell(
                    onTap: () async {
                      final t = await showTimePicker(
                        context: context, 
                        initialTime: TimeOfDay.now(),
                        builder: (context, child) {
                          return Theme(
                            data: ThemeData.dark().copyWith(
                              colorScheme: const ColorScheme.dark(
                                primary: Color(0xFF7F7CFF),
                                onPrimary: Colors.white,
                                surface: Color(0xFF1C1E2B),
                                onSurface: Colors.white,
                              ),
                            ),
                            child: child!,
                          );
                        },
                      );
                      if (t != null) setState(() => _selectedTime = t);
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1C1E2B),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            _selectedTime != null 
                                ? '${_selectedTime!.hour.toString().padLeft(2, '0')}:${_selectedTime!.minute.toString().padLeft(2, '0')}'
                                : '시간 선택',
                            style: TextStyle(color: _selectedTime != null ? Colors.white : Colors.white54, fontSize: 16),
                          ),
                          const Icon(Icons.access_time, color: Colors.white54),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1C1E2B),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: presetsAsync.when(
                      data: (presets) {
                        return DropdownButtonHideUnderline(
                          child: DropdownButton<int>(
                            value: _selectedPresetId,
                            hint: const Text('프리셋 선택', style: TextStyle(color: Colors.white54)),
                            dropdownColor: const Color(0xFF1C1E2B),
                            isExpanded: true,
                            icon: const Icon(Icons.arrow_drop_down, color: Colors.white54),
                            items: presets.map<DropdownMenuItem<int>>((p) {
                              return DropdownMenuItem<int>(
                                value: p['id'],
                                child: Text(p['name'], style: const TextStyle(color: Colors.white)),
                              );
                            }).toList(),
                            onChanged: (val) => setState(() => _selectedPresetId = val),
                          ),
                        );
                      },
                      loading: () => const Center(child: CircularProgressIndicator()),
                      error: (_,__) => const SizedBox(),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: (_selectedTime == null || _selectedPresetId == null) 
                    ? null 
                    : () {
                        final timeStr = '${_selectedTime!.hour.toString().padLeft(2, '0')}:${_selectedTime!.minute.toString().padLeft(2, '0')}';
                        ControlActions.addSchedule(ref, _selectedPresetId!, timeStr);
                        setState(() {
                          _selectedTime = null;
                          _selectedPresetId = null;
                        });
                      },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF7F7CFF),
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  child: const Text('추가', style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
