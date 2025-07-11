/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathMeasure_DEFINED
#define SkPathMeasure_DEFINED

#include "include/core/SkContourMeasure.h"
#include "include/core/SkPoint.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkScalar.h"
#include "include/core/SkTypes.h"
#include "include/private/base/SkDebug.h"

class SkMatrix;
class SkPath;
class SkPathBuilder;

class SK_API SkPathMeasure {
public:
    SkPathMeasure();
    /** Initialize the pathmeasure with the specified path. The parts of the path that are needed
     *  are copied, so the client is free to modify/delete the path after this call.
     *
     *  resScale controls the precision of the measure. values > 1 increase the
     *  precision (and possibly slow down the computation).
     */
    SkPathMeasure(const SkPath& path, bool forceClosed, SkScalar resScale = 1);
    ~SkPathMeasure();

    SkPathMeasure(SkPathMeasure&&) = default;
    SkPathMeasure& operator=(SkPathMeasure&&) = default;

    /** Reset the pathmeasure with the specified path. The parts of the path that are needed
     *  are copied, so the client is free to modify/delete the path after this call..
     */
    void setPath(const SkPath*, bool forceClosed);

    /** Return the total length of the current contour, or 0 if no path
        is associated (e.g. resetPath(null))
    */
    SkScalar getLength();

    /** Pins distance to 0 <= distance <= getLength(), and then computes
        the corresponding position and tangent.
        Returns false if there is no path, or a zero-length path was specified, in which case
        position and tangent are unchanged.
    */
    [[nodiscard]] bool getPosTan(SkScalar distance, SkPoint* position, SkVector* tangent);

    enum MatrixFlags {
        kGetPosition_MatrixFlag     = 0x01,
        kGetTangent_MatrixFlag      = 0x02,
        kGetPosAndTan_MatrixFlag    = kGetPosition_MatrixFlag | kGetTangent_MatrixFlag
    };

    /** Pins distance to 0 <= distance <= getLength(), and then computes
        the corresponding matrix (by calling getPosTan).
        Returns false if there is no path, or a zero-length path was specified, in which case
        matrix is unchanged.
    */
    [[nodiscard]] bool getMatrix(SkScalar distance, SkMatrix* matrix,
                                 MatrixFlags flags = kGetPosAndTan_MatrixFlag);

    /** Given a start and stop distance, return in dst the intervening segment(s).
        If the segment is zero-length, return false, else return true.
        startD and stopD are pinned to legal values (0..getLength()). If startD > stopD
        then return false (and leave dst untouched).
        Begin the segment with a moveTo if startWithMoveTo is true
    */
    bool getSegment(SkScalar startD, SkScalar stopD, SkPathBuilder* dst, bool startWithMoveTo);
#ifdef SK_SUPPORT_MUTABLE_PATHEFFECT
    bool getSegment(SkScalar startD, SkScalar stopD, SkPath* dst, bool startWithMoveTo);
#endif

    /** Return true if the current contour is closed()
    */
    bool isClosed();

    /** Move to the next contour in the path. Return true if one exists, or false if
        we're done with the path.
    */
    bool nextContour();

#ifdef SK_DEBUG
    void    dump();
#endif

    const SkContourMeasure* currentMeasure() const { return fContour.get(); }

private:
    SkContourMeasureIter    fIter;
    sk_sp<SkContourMeasure> fContour;
};

#endif
