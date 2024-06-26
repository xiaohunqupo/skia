// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
#include "tools/fiddle/examples.h"
REG_FIDDLE(Image_BorrowTextureFrom_2, 256, 256, false, 4) {
void draw(SkCanvas* canvas) {
    auto dContext = GrAsDirectContext(canvas->recordingContext());
    if (!dContext) {
       return;
    }

    auto releaseCallback = [](SkImages::ReleaseContext releaseContext) -> void {
        *((int*)releaseContext) += 128;
    };
    int x = 0, y = 0;
    for (auto origin : { kBottomLeft_GrSurfaceOrigin, kTopLeft_GrSurfaceOrigin } ) {
        sk_sp<SkImage> image = SkImages::BorrowTextureFrom(dContext,
                                                           backEndTexture,
                                                           origin,
                                                           kRGBA_8888_SkColorType,
                                                           kOpaque_SkAlphaType,
                                                           nullptr,
                                                           releaseCallback,
                                                           &x);
        canvas->drawImage(image, x, y);
        y += 128;
    }
}
}  // END FIDDLE
