import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'mainpage.dart';

// Pretendard 폰트가 로드 실패할 때를 대비한 시스템 폰트 fallback 목록
const kFontFallback = ['Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', 'sans-serif'];

void main() {
  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    final baseTheme = ThemeData(
      useMaterial3: true,
      fontFamily: 'Pretendard',
      brightness: Brightness.dark,
      scaffoldBackgroundColor: const Color(0xFF11121A),
      colorScheme: const ColorScheme.dark(
        primary: Color(0xFF7F7CFF),
        surface: Color(0xFF181924),
      ),
    );

    return MaterialApp(
      title: 'SENTRYNEX Control.',
      debugShowCheckedModeBanner: false,
      home: const MainPage(),
      // 테마의 모든 TextStyle에 fontFamilyFallback 적용
      theme: baseTheme.copyWith(
        textTheme: _applyFallback(baseTheme.textTheme),
      ),
    );
  }
}

/// 테마의 모든 TextStyle에 fontFamilyFallback을 일괄 적용
TextTheme _applyFallback(TextTheme base) {
  return TextTheme(
    displayLarge: base.displayLarge?.copyWith(fontFamilyFallback: kFontFallback),
    displayMedium: base.displayMedium?.copyWith(fontFamilyFallback: kFontFallback),
    displaySmall: base.displaySmall?.copyWith(fontFamilyFallback: kFontFallback),
    headlineLarge: base.headlineLarge?.copyWith(fontFamilyFallback: kFontFallback),
    headlineMedium: base.headlineMedium?.copyWith(fontFamilyFallback: kFontFallback),
    headlineSmall: base.headlineSmall?.copyWith(fontFamilyFallback: kFontFallback),
    titleLarge: base.titleLarge?.copyWith(fontFamilyFallback: kFontFallback),
    titleMedium: base.titleMedium?.copyWith(fontFamilyFallback: kFontFallback),
    titleSmall: base.titleSmall?.copyWith(fontFamilyFallback: kFontFallback),
    bodyLarge: base.bodyLarge?.copyWith(fontFamilyFallback: kFontFallback),
    bodyMedium: base.bodyMedium?.copyWith(fontFamilyFallback: kFontFallback),
    bodySmall: base.bodySmall?.copyWith(fontFamilyFallback: kFontFallback),
    labelLarge: base.labelLarge?.copyWith(fontFamilyFallback: kFontFallback),
    labelMedium: base.labelMedium?.copyWith(fontFamilyFallback: kFontFallback),
    labelSmall: base.labelSmall?.copyWith(fontFamilyFallback: kFontFallback),
  );
}
