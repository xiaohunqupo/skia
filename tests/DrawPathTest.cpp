/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkBitmap.h"
#include "include/core/SkCanvas.h"
#include "include/core/SkColor.h"
#include "include/core/SkImageInfo.h"
#include "include/core/SkMatrix.h"
#include "include/core/SkPaint.h"
#include "include/core/SkPath.h"
#include "include/core/SkPathBuilder.h"
#include "include/core/SkPathEffect.h"
#include "include/core/SkPathTypes.h"
#include "include/core/SkPathUtils.h"
#include "include/core/SkPoint.h"
#include "include/core/SkRRect.h"
#include "include/core/SkRect.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkScalar.h"
#include "include/core/SkStrokeRec.h"
#include "include/core/SkSurface.h"
#include "include/core/SkTypes.h"
#include "include/effects/SkDashPathEffect.h"
#include "tests/Test.h"

#include <cstdint>

// test that we can draw an aa-rect at coordinates > 32K (bigger than fixedpoint)
static void test_big_aa_rect(skiatest::Reporter* reporter) {
    SkBitmap output;
    SkPMColor pixel[1];
    output.installPixels(SkImageInfo::MakeN32Premul(1, 1), pixel, 4);

    auto surf = SkSurfaces::Raster(SkImageInfo::MakeN32Premul(300, 33300));
    SkCanvas* canvas = surf->getCanvas();

    SkRect r = { 0, 33000, 300, 33300 };
    int x = SkScalarRoundToInt(r.left());
    int y = SkScalarRoundToInt(r.top());

    // check that the pixel in question starts as transparent (by the surface)
    if (surf->readPixels(output, x, y)) {
        REPORTER_ASSERT(reporter, 0 == pixel[0]);
    } else {
        REPORTER_ASSERT(reporter, false, "readPixels failed");
    }

    SkPaint paint;
    paint.setAntiAlias(true);
    paint.setColor(SK_ColorWHITE);

    canvas->drawRect(r, paint);

    // Now check that it is BLACK
    if (surf->readPixels(output, x, y)) {
        // don't know what swizzling PMColor did, but white should always
        // appear the same.
        REPORTER_ASSERT(reporter, 0xFFFFFFFF == pixel[0]);
    } else {
        REPORTER_ASSERT(reporter, false, "readPixels failed");
    }
}

///////////////////////////////////////////////////////////////////////////////

static void moveToH(SkPath* path, const uint32_t raw[]) {
    const float* fptr = (const float*)raw;
    path->moveTo(fptr[0], fptr[1]);
}

static void cubicToH(SkPath* path, const uint32_t raw[]) {
    const float* fptr = (const float*)raw;
    path->cubicTo(fptr[0], fptr[1], fptr[2], fptr[3], fptr[4], fptr[5]);
}

// This used to assert, because we performed a cast (int)(pt[0].fX * scale) to
// arrive at an int (SkFDot6) rather than calling sk_float_round2int. The assert
// was that the initial line-segment produced by the cubic was not monotonically
// going down (i.e. the initial DY was negative). By rounding the floats, we get
// the more proper result.
//
// http://code.google.com/p/chromium/issues/detail?id=131181
//

// we're not calling this test anymore; is that for a reason?

static void test_crbug131181() {
    /*
     fX = 18.8943768,
     fY = 129.121277
     }, {
     fX = 18.8937435,
     fY = 129.121689
     }, {
     fX = 18.8950119,
     fY = 129.120422
     }, {
     fX = 18.5030727,
     fY = 129.13121
     */
    uint32_t data[] = {
        0x419727af, 0x43011f0c, 0x41972663, 0x43011f27,
        0x419728fc, 0x43011ed4, 0x4194064b, 0x43012197
    };

    SkPath path;
    moveToH(&path, &data[0]);
    cubicToH(&path, &data[2]);

    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(640, 480)));

    SkPaint paint;
    paint.setAntiAlias(true);
    surface->getCanvas()->drawPath(path, paint);
}

// This used to assert in debug builds (and crash writing bad memory in release)
// because we overflowed an intermediate value (B coefficient) setting up our
// stepper for the quadratic. Now we bias that value by 1/2 so we don't overflow
static void test_crbug_140803() {
    SkBitmap bm;
    bm.allocN32Pixels(2700, 30*1024);
    SkCanvas canvas(bm);

    SkPaint paint;
    paint.setAntiAlias(true);
    canvas.drawPath(SkPath().moveTo(2762, 20).quadTo(11, 21702, 10, 21706), paint);
}

static void test_crbug_1239558(skiatest::Reporter* reporter) {
    SkBitmap bm;
    bm.allocN32Pixels(256, 256);

    SkCanvas canvas(bm);
    canvas.clear(SK_ColorWHITE);

    // This creates a single cubic where the control points form an extremely skinny, vertical
    // triangle contained within the x=0 column of pixels. Since it is convex (ignoring the leading
    // moveTo's) it uses the convex aaa optimized edge walking algorithm after clipping the path to
    // the device bounds. However, due to fixed-point math while walking these edges, the edge
    // walking evaluates to coords that are very slightly less than 0 (i.e. 0.0012). Both the left
    // and right edges would be out of bounds, but the edge walking is optimized to only clamp the
    // left edge to the left bounds, and the right edge to the right bounds. After this clamping,
    // the left and right edges are no longer sorted. This then led to incorrect behavior in various
    // forms (described below).
    SkPath path;
    path.setFillType(SkPathFillType::kWinding);
    path.moveTo(7.00649e-45f,  2.f);
    path.moveTo(0.0160219f,    7.45063e-09f);
    path.moveTo(192.263f,      8.40779e-44f);
    path.moveTo(7.34684e-40f,  194.25f);
    path.moveTo(2.3449e-38f,   6.01858e-36f);
    path.moveTo(7.34684e-40f,  194.25f);
    path.cubicTo(5.07266e-39f, 56.0488f,
                 0.0119172f,   0.f,
                 7.34684e-40f, 194.25f);

    SkPaint paint;
    paint.setColor(SK_ColorRED);
    paint.setAntiAlias(true);
    // On debug builds, the inverted left/right edges led to a negative coverage that triggered an
    // assert while converting to a uint8 alpha value. On release builds with UBSAN, it would
    // detect a negative left shift when computing the pixel address and crash. On regular release
    // builds it would write a saturate coverage value to pixels that wrapped around to the far edge
    canvas.drawPath(path, paint);

    // UBSAN and debug builds would fail inside the drawPath() call above, but detect the incorrect
    // memory access on release builds so that the test would fail. Given the path, it should only
    // touch pixels with x=0 but the incorrect addressing would wrap to the right edge.
    for (int y = 0; y < 256; ++y) {
        if (bm.getColor(255, y) != SK_ColorWHITE) {
            REPORTER_ASSERT(reporter, false, "drawPath modified incorrect pixels");
            break;
        }
    }
}

// Need to exercise drawing an inverse-path whose bounds intersect the clip,
// but whose edges do not (since its a quad which draws only in the bottom half
// of its bounds).
// In the debug build, we used to assert in this case, until it was fixed.
//
static void test_inversepathwithclip() {
    SkPath path;

    path.moveTo(0, 20);
    path.quadTo(10, 10, 20, 20);
    path.toggleInverseFillType();

    SkPaint paint;

    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(640, 480)));
    SkCanvas* canvas = surface->getCanvas();
    canvas->save();
    canvas->clipRect(SkRect::MakeWH(19, 11));

    paint.setAntiAlias(false);
    canvas->drawPath(path, paint);
    paint.setAntiAlias(true);
    canvas->drawPath(path, paint);

    canvas->restore();

    // Now do the test again, with the path flipped, so we only draw in the
    // top half of our bounds, and have the clip intersect our bounds at the
    // bottom.
    path.reset();   // preserves our filltype
    path.moveTo(0, 10);
    path.quadTo(10, 20, 20, 10);
    canvas->clipRect(SkRect::MakeXYWH(0, 19, 19, 11));

    paint.setAntiAlias(false);
    canvas->drawPath(path, paint);
    paint.setAntiAlias(true);
    canvas->drawPath(path, paint);
}

static void test_bug533() {
    /*
        http://code.google.com/p/skia/issues/detail?id=533
        This particular test/bug only applies to the float case, where the
        coordinates are very large.
     */
    SkPath path;
    path.moveTo(64, 3);
    path.quadTo(-329936, -100000000, 1153, 330003);

    SkPaint paint;
    paint.setAntiAlias(true);

    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(640, 480)));
    surface->getCanvas()->drawPath(path, paint);
}

static void test_crbug_140642() {
    /*
     *  We used to see this construct, and due to rounding as we accumulated
     *  our length, the loop where we apply the phase would run off the end of
     *  the array, since it relied on just -= each interval value, which did not
     *  behave as "expected". Now the code explicitly checks for walking off the
     *  end of that array.

     *  A different (better) fix might be to rewrite dashing to do all of its
     *  length/phase/measure math using double, but this may need to be
     *  coordinated with SkPathMeasure, to be consistent between the two.

     <path stroke="mintcream" stroke-dasharray="27734 35660 2157846850 247"
           stroke-dashoffset="-248.135982067">
     */

    const SkScalar vals[] = { 27734, 35660, 2157846850.0f, 247 };
    auto dontAssert = SkDashPathEffect::Make(vals, -248.135982067f);
}

static void test_crbug_124652() {
    /*
        http://code.google.com/p/chromium/issues/detail?id=124652
        This particular test/bug only applies to the float case, where
        large values can "swamp" small ones.
     */
    SkScalar intervals[2] = {837099584, 33450};
    auto dontAssert = SkDashPathEffect::Make(intervals, -10);
}

static void test_bigcubic() {
    SkPath path;
    path.moveTo(64, 3);
    path.cubicTo(-329936, -100000000, -329936, 100000000, 1153, 330003);

    SkPaint paint;
    paint.setAntiAlias(true);

    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(640, 480)));
    surface->getCanvas()->drawPath(path, paint);
}

// asserts if halfway case is not handled
static void test_halfway() {
    SkPaint paint;
    SkPath path;
    path.moveTo(16365.5f, 1394);
    path.lineTo(16365.5f, 1387.5f);
    path.quadTo(16365.5f, 1385.43f, 16367, 1383.96f);
    path.quadTo(16368.4f, 1382.5f, 16370.5f, 1382.5f);
    path.lineTo(16465.5f, 1382.5f);
    path.quadTo(16467.6f, 1382.5f, 16469, 1383.96f);
    path.quadTo(16470.5f, 1385.43f, 16470.5f, 1387.5f);
    path.lineTo(16470.5f, 1394);
    path.quadTo(16470.5f, 1396.07f, 16469, 1397.54f);
    path.quadTo(16467.6f, 1399, 16465.5f, 1399);
    path.lineTo(16370.5f, 1399);
    path.quadTo(16368.4f, 1399, 16367, 1397.54f);
    path.quadTo(16365.5f, 1396.07f, 16365.5f, 1394);
    path.close();
    SkPath p2;
    SkMatrix m;
    m.reset();
    m.postTranslate(0.001f, 0.001f);
    path.transform(m, &p2);

    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(640, 480)));
    SkCanvas* canvas = surface->getCanvas();
    canvas->translate(-16366, -1383);
    canvas->drawPath(p2, paint);

    m.reset();
    m.postTranslate(-0.001f, -0.001f);
    path.transform(m, &p2);
    canvas->drawPath(p2, paint);

    m.reset();
    path.transform(m, &p2);
    canvas->drawPath(p2, paint);
}

// we used to assert if the bounds of the device (clip) was larger than 32K
// even when the path itself was smaller. We just draw and hope in the debug
// version to not assert.
static void test_giantaa() {
    const int W = 400;
    const int H = 400;
    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(33000, 10)));

    SkPaint paint;
    paint.setAntiAlias(true);
    SkPath path;
    path.addOval(SkRect::MakeXYWH(-10, -10, 20 + W, 20 + H));
    surface->getCanvas()->drawPath(path, paint);
}

// Extremely large path_length/dash_length ratios may cause infinite looping
// in SkDashPathEffect::filterPath() due to single precision rounding.
// The test is quite expensive, but it should get much faster after the fix
// for http://crbug.com/165432 goes in.
static void test_infinite_dash(skiatest::Reporter* reporter) {
    SkPath path;
    path.moveTo(0, 0);
    path.lineTo(5000000, 0);

    SkScalar intervals[] = { 0.2f, 0.2f };
    sk_sp<SkPathEffect> dash(SkDashPathEffect::Make(intervals, 0));

    SkPaint paint;
    paint.setStyle(SkPaint::kStroke_Style);
    paint.setPathEffect(dash);

    (void)skpathutils::FillPathWithPaint(path, paint);
    // If we reach this, we passed.
    REPORTER_ASSERT(reporter, true);
}

// http://crbug.com/165432
// Limit extreme dash path effects to avoid exhausting the system memory.
static void test_crbug_165432(skiatest::Reporter* reporter) {
    SkPath path;
    path.moveTo(0, 0);
    path.lineTo(10000000, 0);

    SkScalar intervals[] = { 0.5f, 0.5f };
    sk_sp<SkPathEffect> dash(SkDashPathEffect::Make(intervals, 0));

    SkPaint paint;
    paint.setStyle(SkPaint::kStroke_Style);
    paint.setPathEffect(dash);

    SkPathBuilder filteredPath;
    SkStrokeRec rec(paint);
    REPORTER_ASSERT(reporter, !dash->filterPath(&filteredPath, path, &rec));
    REPORTER_ASSERT(reporter, filteredPath.isEmpty());
}

// http://crbug.com/472147
// This is a simplified version from the bug. RRect radii not properly scaled.
static void test_crbug_472147_simple(skiatest::Reporter* reporter) {
    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(1000, 1000)));
    SkCanvas* canvas = surface->getCanvas();
    SkPaint p;
    SkRect r = SkRect::MakeLTRB(-246.0f, 33.0f, 848.0f, 33554464.0f);
    SkVector radii[4] = {
        { 13.0f, 8.0f }, { 170.0f, 2.0 }, { 256.0f, 33554430.0f }, { 120.0f, 5.0f }
    };
    SkRRect rr;
    rr.setRectRadii(r, radii);
    canvas->drawRRect(rr, p);
}

// http://crbug.com/472147
// RRect radii not properly scaled.
static void test_crbug_472147_actual(skiatest::Reporter* reporter) {
    auto surface(SkSurfaces::Raster(SkImageInfo::MakeN32Premul(1000, 1000)));
    SkCanvas* canvas = surface->getCanvas();
    SkPaint p;
    SkRect r = SkRect::MakeLTRB(-246.0f, 33.0f, 848.0f, 33554464.0f);
    SkVector radii[4] = {
        { 13.0f, 8.0f }, { 170.0f, 2.0 }, { 256.0f, 33554430.0f }, { 120.0f, 5.0f }
    };
    SkRRect rr;
    rr.setRectRadii(r, radii);
    canvas->clipRRect(rr);

    SkRect r2 = SkRect::MakeLTRB(0, 33, 1102, 33554464);
    canvas->drawRect(r2, p);
}

DEF_TEST(DrawPath, reporter) {
    test_giantaa();
    test_bug533();
    test_bigcubic();
    test_crbug_124652();
    test_crbug_140642();
    test_crbug_140803();
    test_inversepathwithclip();
    // why?
    if ((false)) test_crbug131181();
    test_infinite_dash(reporter);
    test_crbug_165432(reporter);
    test_crbug_472147_simple(reporter);
    test_crbug_472147_actual(reporter);
    test_crbug_1239558(reporter);
    test_big_aa_rect(reporter);
    test_halfway();
}
