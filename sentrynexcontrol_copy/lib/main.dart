import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'mainpage.dart';

void main() {
  runApp(const ProviderScope(child: MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SENTRYNEX Control.',
      debugShowCheckedModeBanner: false,
      home: const MainPage(),
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Pretendard',
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF11121A),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF7F7CFF),
          surface: Color(0xFF181924),
        ),
      ),
    );
  }
}
