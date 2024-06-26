// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
#include "tools/fiddle/examples.h"
REG_FIDDLE(Bitmap_getBounds_2, 256, 256, false, 3) {
void draw(SkCanvas* canvas) {
    SkIRect bounds;
    source.getBounds(&bounds);
    bounds.inset(100, 100);
    SkBitmap bitmap;
    source.extractSubset(&bitmap, bounds);
    canvas->scale(.5f, .5f);
    canvas->drawImage(bitmap.asImage(), 10, 10);
}
}  // END FIDDLE
