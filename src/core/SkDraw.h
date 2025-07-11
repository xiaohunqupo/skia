
/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */


#ifndef SkDraw_DEFINED
#define SkDraw_DEFINED

#include "include/core/SkCanvas.h"
#include "include/core/SkColor.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkSamplingOptions.h"
#include "include/core/SkSpan.h"
#include "src/base/SkZip.h"
#include "src/core/SkDrawBase.h"

class SkArenaAlloc;
class SkBitmap;
class SkBlender;
class SkDevice;
class SkGlyph;
class SkGlyphRunListPainterCPU;
class SkMatrix;
class SkPaint;
class SkVertices;
namespace sktext { class GlyphRunList; }
struct SkMask;
struct SkPoint3;
struct SkPoint;
struct SkRSXform;
struct SkRect;


// defaults to use SkBlitter::Choose()
class SkDraw : public SkDrawBase {
public:
    SkDraw();

    /* If dstOrNull is null, computes a dst by mapping the bitmap's bounds through the matrix. */
    void    drawBitmap(const SkBitmap&, const SkMatrix&, const SkRect* dstOrNull,
                       const SkSamplingOptions&, const SkPaint&) const override;
    void    drawSprite(const SkBitmap&, int x, int y, const SkPaint&) const;
    void    drawGlyphRunList(SkCanvas* canvas,
                             SkGlyphRunListPainterCPU* glyphPainter,
                             const sktext::GlyphRunList& glyphRunList,
                             const SkPaint& paint) const;

    void paintMasks(SkZip<const SkGlyph*, SkPoint> accepted, const SkPaint& paint) const override;

    void drawPoints(SkCanvas::PointMode, SkSpan<const SkPoint>, const SkPaint&, SkDevice*) const;
    /* If skipColorXform, skips color conversion when assigning per-vertex colors */
    void drawVertices(const SkVertices*,
                      sk_sp<SkBlender>,
                      const SkPaint&,
                      bool skipColorXform) const;
    void drawAtlas(SkSpan<const SkRSXform>, SkSpan<const SkRect>, SkSpan<const SkColor>,
                   sk_sp<SkBlender>, const SkPaint&);

    void drawDevMask(const SkMask& mask, const SkPaint&, const SkMatrix*) const;
    void drawBitmapAsMask(const SkBitmap&, const SkSamplingOptions&, const SkPaint&,
                          const SkMatrix* paintMatrix) const;

private:
    void drawFixedVertices(const SkVertices* vertices,
                           sk_sp<SkBlender> blender,
                           const SkPaint& paint,
                           const SkMatrix& ctmInverse,
                           const SkPoint* dev2,
                           const SkPoint3* dev3,
                           SkArenaAlloc* outerAlloc,
                           bool skipColorXform) const;
};

#endif
