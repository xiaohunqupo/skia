/*
 * Copyright 2017 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/gpu/ganesh/GrOnFlushResourceProvider.h"

#include "include/gpu/ganesh/GrContextOptions.h"
#include "include/gpu/ganesh/GrDirectContext.h"
#include "include/gpu/ganesh/GrRecordingContext.h"
#include "include/private/base/SkAssert.h"
#include "src/gpu/ganesh/GrDirectContextPriv.h"
#include "src/gpu/ganesh/GrDrawingManager.h"
#include "src/gpu/ganesh/GrRecordingContextPriv.h"
#include "src/gpu/ganesh/GrSurfaceProxy.h"
#include "src/gpu/ganesh/GrSurfaceProxyPriv.h"

bool GrOnFlushResourceProvider::instantiateProxy(GrSurfaceProxy* proxy) {
    SkASSERT(proxy->canSkipResourceAllocator());

    // TODO: this class should probably just get a GrDirectContext
    auto direct = fDrawingMgr->getContext()->asDirectContext();
    if (!direct) {
        return false;
    }

    auto resourceProvider = direct->priv().resourceProvider();

    if (proxy->isLazy()) {
        return proxy->priv().doLazyInstantiation(resourceProvider);
    }

    return proxy->instantiate(resourceProvider);
}

const GrCaps* GrOnFlushResourceProvider::caps() const {
    return fDrawingMgr->getContext()->priv().caps();
}

#if defined(GPU_TEST_UTILS)
bool GrOnFlushResourceProvider::failFlushTimeCallbacks() const {
    return fDrawingMgr->getContext()->priv().options().fFailFlushTimeCallbacks;
}
#endif
