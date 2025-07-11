/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkTypes.h"

#ifdef SK_SUPPORT_PDF

#include "include/core/SkBlendMode.h"
#include "include/core/SkCanvas.h"
#include "include/core/SkColor.h"
#include "include/core/SkDocument.h"
#include "include/core/SkPaint.h"
#include "include/core/SkRect.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkStream.h"
#include "include/docs/SkPDFDocument.h"
#include "include/docs/SkPDFJpegHelpers.h"
#include "include/private/base/SkCPUTypes.h"
#include "tests/Test.h"

static void run_test(SkWStream* out, SkBlendMode mode, U8CPU alpha) {
    auto pdfDoc = SkPDF::MakeDocument(out, SkPDF::JPEG::MetadataWithCallbacks());
    SkCanvas* c = pdfDoc->beginPage(612.0f, 792.0f);
    SkPaint black;
    SkPaint background;
    background.setColor(SK_ColorWHITE);
    background.setAlpha(alpha);
    background.setBlendMode(mode);
    c->drawRect(SkRect::MakeWH(612.0f, 792.0f), background);
    c->drawRect(SkRect::MakeXYWH(36.0f, 36.0f, 9.0f, 9.0f), black);
    c->drawRect(SkRect::MakeXYWH(72.0f, 72.0f, 468.0f, 648.0f), background);
    c->drawRect(SkRect::MakeXYWH(108.0f, 108.0f, 9.0f, 9.0f), black);
    pdfDoc->close();
}

// http://crbug.com/473572
DEF_TEST(SkPDF_OpaqueSrcModeToSrcOver, r) {
    REQUIRE_PDF_DOCUMENT(SkPDF_OpaqueSrcModeToSrcOver, r);
    SkDynamicMemoryWStream srcMode;
    SkDynamicMemoryWStream srcOverMode;

    U8CPU alpha = SK_AlphaOPAQUE;
    run_test(&srcMode, SkBlendMode::kSrc, alpha);
    run_test(&srcOverMode, SkBlendMode::kSrcOver, alpha);
    REPORTER_ASSERT(r, srcMode.bytesWritten() == srcOverMode.bytesWritten());
    // The two PDFs should be equal because they have an opaque alpha.

    srcMode.reset();
    srcOverMode.reset();

    alpha = 0x80;
    run_test(&srcMode, SkBlendMode::kSrc, alpha);
    run_test(&srcOverMode, SkBlendMode::kSrcOver, alpha);
    REPORTER_ASSERT(r, srcMode.bytesWritten() > srcOverMode.bytesWritten());
    // The two PDFs should not be equal because they have a non-opaque alpha.
}

#endif // SK_SUPPORT_PDF
