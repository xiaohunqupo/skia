/*
 * Copyright 2023 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/gpu/ganesh/image/SkImage_LazyTexture.h"

#include "include/core/SkColorSpace.h"
#include "include/core/SkImage.h"
#include "include/core/SkImageGenerator.h"
#include "include/core/SkImageInfo.h"
#include "include/core/SkPixmap.h"
#include "include/gpu/GpuTypes.h"
#include "include/gpu/ganesh/GrDirectContext.h"
#include "include/gpu/ganesh/SkImageGanesh.h"
#include "include/private/gpu/ganesh/GrTextureGenerator.h"  // IWYU pragma: keep
#include "src/gpu/ganesh/GrColorInfo.h"
#include "src/gpu/ganesh/GrDirectContextPriv.h"
#include "src/gpu/ganesh/GrSurfaceProxyView.h"
#include "src/gpu/ganesh/SkGaneshRecorder.h"
#include "src/gpu/ganesh/SkGr.h"
#include "src/gpu/ganesh/SurfaceContext.h"
#include "src/gpu/ganesh/image/GrImageUtils.h"

#include <cstddef>
#include <memory>
#include <utility>

enum class GrColorType;

sk_sp<SkImage> SkImage_LazyTexture::onMakeSubset(SkRecorder* recorder,
                                                 const SkIRect& subset,
                                                 RequiredProperties props) const {
    sk_sp<SkImage> pixels;
    if (auto gRecorder = AsGaneshRecorder(recorder)) {
        if (auto direct = gRecorder->directContext()) {
            pixels = SkImages::TextureFromImage(direct, this);
        }
    }
    if (!pixels) {
        pixels = this->makeRasterImage(nullptr);
    }
    if (!pixels) {
        return nullptr;
    }
    return pixels->makeSubset(recorder, subset, props);
}

bool SkImage_LazyTexture::readPixelsProxy(GrDirectContext* ctx, const SkPixmap& pixmap) const {
    if (!ctx) {
        return false;
    }
    GrSurfaceProxyView view = skgpu::ganesh::LockTextureProxyView(
            ctx, this, GrImageTexGenPolicy::kDraw, skgpu::Mipmapped::kNo);

    if (!view) {
        return false;
    }

    GrColorType ct = skgpu::ganesh::ColorTypeOfLockTextureProxy(ctx->priv().caps(),
                                                                this->colorType());
    GrColorInfo colorInfo(ct, this->alphaType(), this->refColorSpace());
    auto sContext = ctx->priv().makeSC(std::move(view), colorInfo);
    if (!sContext) {
        return false;
    }
    size_t rowBytes = this->imageInfo().minRowBytes();
    return sContext->readPixels(ctx, {this->imageInfo(), pixmap.writable_addr(), rowBytes}, {0, 0});
}

namespace SkImages {
sk_sp<SkImage> DeferredFromTextureGenerator(std::unique_ptr<GrTextureGenerator> generator) {
    SkImage_Lazy::Validator validator(
            SharedGenerator::Make(std::move(generator)), nullptr, nullptr);

    return validator ? sk_make_sp<SkImage_LazyTexture>(&validator) : nullptr;
}
}
