import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:window_manager/window_manager.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await windowManager.ensureInitialized();
  runApp(const GazePdfApp());
}

class GazePdfApp extends StatelessWidget {
  const GazePdfApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gaze PDF Reader',
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFE2E8F0),
      ),
      home: const GazeReaderScreen(),
    );
  }
}

// ---------------------------------------------------------
// PDF MODELS
// ---------------------------------------------------------

class PdfPage {
  final double width;
  final double height;
  final List<PdfText> textBlocks;
  final List<PdfImage> images;

  PdfPage({
    required this.width,
    required this.height,
    required this.textBlocks,
    required this.images,
  });
}

class PdfLine {
  final Rect bbox;
  final String text;

  PdfLine({
    required this.bbox,
    required this.text,
  });
}

class PdfText {
  final Rect bbox;
  final String text;
  final String font;
  final double size;
  final int flags;
  final List<PdfLine> lines;

  PdfText({
    required this.bbox,
    required this.text,
    required this.font,
    required this.size,
    required this.flags,
    required this.lines,
  });
}

class PdfImage {
  final Rect bbox;
  final ui.Image image;

  PdfImage({
    required this.bbox,
    required this.image,
  });
}

// ---------------------------------------------------------
// MAIN SCREEN & WEBSOCKET STATE
// ---------------------------------------------------------

class GazeReaderScreen extends StatefulWidget {
  const GazeReaderScreen({super.key});

  @override
  State<GazeReaderScreen> createState() => _GazeReaderScreenState();
}

class _GazeReaderScreenState extends State<GazeReaderScreen> {
  PdfPage? _page;
  late WebSocketChannel _channel;
  final GlobalKey _canvasKey = GlobalKey();

  double _gazeX = -100;
  double _gazeY = -100;
  bool _isGazeLocked = true;
  
  static const double pdfScale = 1.2;

  @override
  void initState() {
    super.initState();
    _loadPdfData();
    _initWebSocket();
  }

  Future<void> _loadPdfData() async {
    final jsonString = await rootBundle.loadString('assets/turtles_parsed.json');
    final data = jsonDecode(jsonString);
    final pageData = data['pages'][0];

    List<PdfText> texts = [];
    for (var b in pageData['text_blocks']) {
      final List<PdfLine> lines = [];
      final rawLines = b['lines'] as List<dynamic>?;
      if (rawLines != null) {
        for (var l in rawLines) {
          lines.add(PdfLine(
            bbox: Rect.fromLTRB(
              l['bbox'][0].toDouble(),
              l['bbox'][1].toDouble(),
              l['bbox'][2].toDouble(),
              l['bbox'][3].toDouble(),
            ),
            text: l['text'],
          ));
        }
      }

      texts.add(PdfText(
        bbox: Rect.fromLTRB(
          b['bbox'][0].toDouble(),
          b['bbox'][1].toDouble(),
          b['bbox'][2].toDouble(),
          b['bbox'][3].toDouble(),
        ),
        text: b['text'],
        font: b['font'] ?? 'Arial',
        size: b['size'].toDouble(),
        flags: b['flags'] ?? 0,
        lines: lines,
      ));
    }

    List<PdfImage> images = [];
    for (var i in pageData['images']) {
      final Uint8List bytes = base64Decode(i['data']);
      final ui.Codec codec = await ui.instantiateImageCodec(bytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      
      images.add(PdfImage(
        bbox: Rect.fromLTRB(
          i['bbox'][0].toDouble(),
          i['bbox'][1].toDouble(),
          i['bbox'][2].toDouble(),
          i['bbox'][3].toDouble(),
        ),
        image: frameInfo.image,
      ));
    }

    setState(() {
      _page = PdfPage(
        width: pageData['width'].toDouble(),
        height: pageData['height'].toDouble(),
        textBlocks: texts,
        images: images,
      );
    });
  }

  void _initWebSocket() {
    _channel = WebSocketChannel.connect(
      Uri.parse('ws://localhost:8765'),
    );

    _channel.stream.listen((message) async {
      final gazeData = jsonDecode(message);

      if (!mounted) return;
      final RenderBox? box =
          _canvasKey.currentContext?.findRenderObject() as RenderBox?;
      if (box == null) return;

      final windowPosition = await windowManager.getPosition();
      if (!mounted) return;

      final windowRelative = Offset(
        gazeData['x'].toDouble() - windowPosition.dx,
        gazeData['y'].toDouble() - windowPosition.dy,
      );

      final localOffset = box.globalToLocal(windowRelative);

      setState(() {
        _isGazeLocked = gazeData['locked'];
        // _gazeX = localOffset.dx / pdfScale;
        _gazeX = 0; // Lock gaze X to a fixed value for testing
        _gazeY = localOffset.dy / pdfScale;
      });
    }, onError: (error) {
      debugPrint("WebSocket Error: $error");
    });
  }

  @override
  void dispose() {
    _channel.sink.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_page == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Center(
        child: Container(
          width: _page!.width * pdfScale,
          height: _page!.height * pdfScale,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(8),
            boxShadow: const [
              BoxShadow(
                color: Colors.black12,
                blurRadius: 25,
                offset: Offset(0, 10),
              )
            ],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: CustomPaint(
              key: _canvasKey,
              size: Size(_page!.width * pdfScale, _page!.height * pdfScale),
              painter: PdfCanvasPainter(
                page: _page!,
                gazeY: _isGazeLocked ? _gazeY : null,
                gazeRawOffset: Offset(_gazeX * pdfScale, _gazeY * pdfScale),
                scale: pdfScale,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------
// CUSTOM PAINTER (THE RENDER ENGINE)
// ---------------------------------------------------------

class PdfCanvasPainter extends CustomPainter {
  final PdfPage page;
  final double? gazeY;
  final Offset? gazeRawOffset;
  final double scale;

  PdfCanvasPainter({
    required this.page,
    required this.gazeY,
    required this.gazeRawOffset,
    required this.scale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Draw Images
    final paint = Paint();
    for (var img in page.images) {
      final destRect = Rect.fromLTRB(
        img.bbox.left * scale, img.bbox.top * scale,
        img.bbox.right * scale, img.bbox.bottom * scale,
      );
      final srcRect = Rect.fromLTWH(0, 0, img.image.width.toDouble(), img.image.height.toDouble());
      canvas.drawImageRect(img.image, srcRect, destRect, paint);
    }

    // 2. Determine which SINGLE paragraph is closest to the gaze
    PdfText? hoveredBlock;
    if (gazeY != null) {
      double minDistance = double.infinity;
      for (var block in page.textBlocks) {
        final blockCenterY = (block.bbox.top + block.bbox.bottom) / 2;
        final dist = (gazeY! - blockCenterY).abs();
        
        if (dist < 60 && dist < minDistance) {
          minDistance = dist;
          hoveredBlock = block;
        }
      }
    }

    // 3. Render all standard text blocks
    for (var textBlock in page.textBlocks) {
      final bool isHovered = (textBlock == hoveredBlock);
      if (isHovered) continue;

      final bool fontIsBold = textBlock.font.contains('Bold') || (textBlock.flags & 16) != 0;
      final String fontFamily = textBlock.font.contains('Times') ? 'Times New Roman' : 'Arial';

      if (textBlock.lines.isNotEmpty) {
        for (var line in textBlock.lines) {
          final lineSpan = TextSpan(
            text: line.text,
            style: TextStyle(
              color: const Color(0xFF64748B),
              fontSize: textBlock.size * scale,
              fontWeight: fontIsBold ? FontWeight.bold : FontWeight.normal,
              height: 1.0,
              fontFamily: fontFamily,
            ),
          );

          final linePainter = TextPainter(
            text: lineSpan,
            textDirection: TextDirection.ltr,
          );

          // Layout with NO constraints so it never wraps.
          linePainter.layout();

          // Calculate the exact physical width the PDF intended for this line
          final double expectedWidth = (line.bbox.right - line.bbox.left) * scale;
          
          canvas.save();
          canvas.translate(line.bbox.left * scale, line.bbox.top * scale);

          // If Flutter renders the line wider than the PDF intended, 
          // horizontally scale (squish) it down instead of wrapping it.
          if (linePainter.width > expectedWidth && expectedWidth > 0) {
            canvas.scale(expectedWidth / linePainter.width, 1.0);
          }

          // Paint at Offset.zero since we already translated the canvas
          linePainter.paint(canvas, Offset.zero);
          canvas.restore();
        }
      } else {
        // Fallback for data without per-line bboxes
        final textSpan = TextSpan(
          text: textBlock.text,
          style: TextStyle(
            color: const Color(0xFF64748B),
            fontSize: textBlock.size * scale,
            fontWeight: fontIsBold ? FontWeight.bold : FontWeight.normal,
            height: 1.0,
            fontFamily: fontFamily,
          ),
        );

        final textPainter = TextPainter(
          text: textSpan,
          textDirection: TextDirection.ltr,
        );

        // Limit the width strictly to its parsed bounding box
        double layoutWidth = (textBlock.bbox.right - textBlock.bbox.left) * scale;
        textPainter.layout(maxWidth: layoutWidth);
        textPainter.paint(
          canvas,
          Offset(textBlock.bbox.left * scale, textBlock.bbox.top * scale),
        );
      }
    }

    // 4. Render the single magnified block using dynamic constraints
    if (hoveredBlock != null) {
      const double magnification = 1.4;

      final textSpan = TextSpan(
        // NEWLINE FIX: Strip out the hardcoded PDF newlines so Flutter handles the wrapping dynamically
        text: hoveredBlock.text.replaceAll('\n', ' '),
        style: TextStyle(
          color: const Color(0xFF0F172A),
          fontSize: (hoveredBlock.size * scale) * magnification,
          fontWeight: FontWeight.bold,
          height: 1.0,
          fontFamily: hoveredBlock.font.contains('Times') ? 'Times New Roman' : 'Arial',
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );

      final originalBlockWidth = (hoveredBlock.bbox.right - hoveredBlock.bbox.left) * scale;
      var layoutWidth = originalBlockWidth * magnification * 1.05;

      final maxAvailableWidth = size.width - (hoveredBlock.bbox.left * scale) - 36;
      if (layoutWidth > maxAvailableWidth) {
        layoutWidth = maxAvailableWidth;
      }

      textPainter.layout(maxWidth: layoutWidth);

      final renderOffset = Offset(
        hoveredBlock.bbox.left * scale,
        (hoveredBlock.bbox.top * scale) - 10,
      );

      var bgRect = Rect.fromLTWH(
        renderOffset.dx - 12,
        renderOffset.dy - 12,
        textPainter.width + 24,
        textPainter.height + 24,
      );

      if (bgRect.right > size.width - 12) {
        bgRect = Rect.fromLTRB(
          bgRect.left,
          bgRect.top,
          size.width - 12,
          bgRect.bottom,
        );
      }

      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(8)),
        Paint()
          ..color = Colors.black.withOpacity(0.1)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6),
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(bgRect, const Radius.circular(8)),
        Paint()..color = Colors.white,
      );

      canvas.save();
      canvas.clipRRect(RRect.fromRectAndRadius(bgRect, const Radius.circular(8)));
      textPainter.paint(canvas, renderOffset);
      canvas.restore();
    }

    // 5. Draw Gaze Cursor Overlay
    if (gazeRawOffset != null) {
      canvas.drawCircle(gazeRawOffset!, 8, Paint()..color = const Color(0x40EF4444)..style = PaintingStyle.fill);
      canvas.drawCircle(gazeRawOffset!, 8, Paint()..color = const Color(0xFFEF4444)..style = PaintingStyle.stroke..strokeWidth = 2.0);
    }
  }

  @override
  bool shouldRepaint(covariant PdfCanvasPainter oldDelegate) {
    return oldDelegate.gazeY != gazeY || oldDelegate.gazeRawOffset != gazeRawOffset;
  }
}