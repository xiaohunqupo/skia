/*
 * Copyright 2010 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkColor.h"
#include "include/core/SkPath.h"
#include "include/core/SkPathTypes.h"
#include "include/core/SkRect.h"
#include "include/core/SkScalar.h"
#include "include/core/SkTypes.h"
#include "src/core/SkBlitter.h"
#include "src/core/SkScan.h"
#include "tests/Test.h"

#include <cstdint>

struct FakeBlitter : public SkBlitter {
    FakeBlitter()
        : m_blitCount(0) { }

    void blitH(int x, int y, int width) override {
        m_blitCount++;
    }

    void blitAntiH(int x, int y, const SkAlpha antialias[], const int16_t runs[]) override {
      SkDEBUGFAIL("blitAntiH not implemented");
    }

    int m_blitCount;
};

// skbug.com/40031085
// Lines which is not clipped by boundary based clipping,
// but skipped after tessellation, should be cleared by the blitter.
DEF_TEST(FillPathInverse, reporter) {
    FakeBlitter blitter;
    SkIRect clip;
    SkPath path;
    int height = 100;
    int width  = 200;
    int expected_lines = 5;
    clip.setLTRB(0, height - expected_lines, width, height);
    path.moveTo(0.0f, 0.0f)
        .quadTo(SkIntToScalar(width/2), SkIntToScalar(height),
              SkIntToScalar(width), 0.0f)
        .close()
        .setFillType(SkPathFillType::kInverseWinding);
    SkRegion rgn(clip);
    SkScan::FillPath(path, rgn, &blitter);

    REPORTER_ASSERT(reporter, blitter.m_blitCount == expected_lines);
}
