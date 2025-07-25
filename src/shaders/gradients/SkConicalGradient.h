/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkConicalGradient_DEFINED
#define SkConicalGradient_DEFINED

#include "include/core/SkFlattenable.h"
#include "include/core/SkPoint.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkScalar.h"
#include "src/shaders/gradients/SkGradientBaseShader.h"

#include <optional>

class SkArenaAlloc;
class SkMatrix;
class SkRasterPipeline;
class SkReadBuffer;
class SkShader;
class SkWriteBuffer;

// Please see https://skia.org/dev/design/conical for how our shader works.
class SkConicalGradient final : public SkGradientBaseShader {
public:
    // See https://skia.org/dev/design/conical for what focal data means and how our shader works.
    // We make it public so the GPU shader can also use it.
    struct FocalData {
        SkScalar fR1;      // r1 after mapping focal point to (0, 0)
        SkScalar fFocalX;  // f
        bool fIsSwapped;   // whether we swapped r0, r1

        // The input r0, r1 are the radii when we map centers to {(0, 0), (1, 0)}.
        // We'll post concat matrix with our transformation matrix that maps focal point to (0, 0).
        // Returns true if the set succeeded
        bool set(SkScalar r0, SkScalar r1, SkMatrix* matrix);

        // Whether the focal point (0, 0) is on the end circle with center (1, 0) and radius r1. If
        // this is true, it's as if an aircraft is flying at Mach 1 and all circles (soundwaves)
        // will go through the focal point (aircraft). In our previous implementations, this was
        // known as the edge case where the inside circle touches the outside circle (on the focal
        // point). If we were to solve for t bruteforcely using a quadratic equation, this case
        // implies that the quadratic equation degenerates to a linear equation.
        bool isFocalOnCircle() const { return SkScalarNearlyZero(1 - fR1); }

        bool isSwapped() const { return fIsSwapped; }
        bool isWellBehaved() const { return !this->isFocalOnCircle() && fR1 > 1; }
        bool isNativelyFocal() const { return SkScalarNearlyZero(fFocalX); }
    };

    enum class Type { kRadial, kStrip, kFocal };

    static std::optional<SkMatrix> MapToUnitX(const SkPoint& startCenter,
                                              const SkPoint& endCenter);

    static sk_sp<SkShader> Create(const SkPoint& start,
                                  SkScalar startRadius,
                                  const SkPoint& end,
                                  SkScalar endRadius,
                                  const Descriptor&,
                                  const SkMatrix* localMatrix);

    GradientType asGradient(GradientInfo* info, SkMatrix* localMatrix) const override;
    bool isOpaque() const override;

    SkScalar getCenterX1() const { return SkPoint::Distance(fCenter1, fCenter2); }
    SkScalar getStartRadius() const { return fRadius1; }
    SkScalar getDiffRadius() const { return fRadius2 - fRadius1; }
    const SkPoint& getStartCenter() const { return fCenter1; }
    const SkPoint& getEndCenter() const { return fCenter2; }
    SkScalar getEndRadius() const { return fRadius2; }

    Type getType() const { return fType; }
    const FocalData& getFocalData() const { return fFocalData; }

    SkConicalGradient(const SkPoint& c0,
                      SkScalar r0,
                      const SkPoint& c1,
                      SkScalar r1,
                      const Descriptor&,
                      Type,
                      const SkMatrix&,
                      const FocalData&);

protected:
    void flatten(SkWriteBuffer& buffer) const override;

    void appendGradientStages(SkArenaAlloc* alloc,
                              SkRasterPipeline* tPipeline,
                              SkRasterPipeline* postPipeline) const override;

private:
    friend void ::SkRegisterConicalGradientShaderFlattenable();
    SK_FLATTENABLE_HOOKS(SkConicalGradient)

    SkPoint fCenter1;
    SkPoint fCenter2;
    SkScalar fRadius1;
    SkScalar fRadius2;
    Type fType;

    FocalData fFocalData;
};

#endif
