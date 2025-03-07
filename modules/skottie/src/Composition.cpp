/*
 * Copyright 2019 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "modules/skottie/src/Composition.h"

#include "include/core/SkString.h"
#include "include/private/base/SkTo.h"
#include "modules/jsonreader/SkJSONReader.h"
#include "modules/skottie/include/Skottie.h"
#include "modules/skottie/src/Camera.h"
#include "modules/skottie/src/SkottieJson.h"
#include "modules/skottie/src/SkottiePriv.h"
#include "modules/sksg/include/SkSGGroup.h"
#include "modules/sksg/include/SkSGRenderNode.h"
#include "modules/sksg/include/SkSGTransform.h"

#include <algorithm>
#include <utility>

namespace skottie {
namespace internal {

AnimationBuilder::ScopedAssetRef::ScopedAssetRef(const AnimationBuilder* abuilder,
                                                 const skjson::ObjectValue& jlayer) {
    const auto refId = ParseDefault<SkString>(jlayer["refId"], SkString());
    if (refId.isEmpty()) {
        abuilder->log(Logger::Level::kError, nullptr, "Layer missing refId.");
        return;
    }

    const auto* asset_info = abuilder->fAssets.find(refId);
    if (!asset_info) {
        abuilder->log(Logger::Level::kError, nullptr, "Asset not found: '%s'.", refId.c_str());
        return;
    }

    if (asset_info->fIsAttaching) {
        abuilder->log(Logger::Level::kError, nullptr,
                  "Asset cycle detected for: '%s'", refId.c_str());
        return;
    }

    asset_info->fIsAttaching = true;

    fInfo = asset_info;
}

CompositionBuilder::CompositionBuilder(const AnimationBuilder& abuilder,
                                       const SkSize& size,
                                       const skjson::ObjectValue& jcomp)
    : fSize(size)
{
    int camera_builder_index = -1;

    // Prepare layer builders.
    if (const skjson::ArrayValue* jlayers = jcomp["layers"]) {
        fLayerBuilders.reserve(SkToInt(jlayers->size()));
        for (const skjson::ObjectValue* jlayer : *jlayers) {
            if (!jlayer) continue;

            const auto  lbuilder_index = fLayerBuilders.size();
            fLayerBuilders.emplace_back(*jlayer, fSize);
            const auto& lbuilder = fLayerBuilders.back();

            fLayerIndexMap.set(lbuilder.index(), lbuilder_index);

            // Keep track of the camera builder.
            if (lbuilder.isCamera()) {
                // We only support one (first) camera for now.
                if (camera_builder_index < 0) {
                    camera_builder_index = SkToInt(lbuilder_index);
                } else {
                    abuilder.log(Logger::Level::kWarning, jlayer,
                                 "Ignoring duplicate camera layer.");
                }
            }
        }
    }

    // Attach a camera transform upfront, if needed (required to build
    // all other 3D transform chains).
    if (camera_builder_index >= 0) {
        // Explicit camera.
        fCameraTransform = fLayerBuilders[camera_builder_index].buildTransform(abuilder, this);
    } else if (ParseDefault<int>(jcomp["ddd"], 0) && !fSize.isEmpty()) {
        // Default/implicit camera when 3D layers are present.
        fCameraTransform = CameraAdaper::DefaultCameraTransform(fSize);
    }
}

CompositionBuilder::~CompositionBuilder() = default;

LayerBuilder* CompositionBuilder::layerBuilder(int layer_index) {
    if (layer_index < 0) {
        return nullptr;
    }

    if (const auto* idx = fLayerIndexMap.find(layer_index)) {
        return &fLayerBuilders[SkToInt(*idx)];
    }

    return nullptr;
}

sk_sp<sksg::RenderNode> CompositionBuilder::layerContent(const AnimationBuilder& abuilder,
                                                         int layer_index) {
    if (auto* lbuilder = this->layerBuilder(layer_index)) {
        return lbuilder->getContentTree(abuilder, this);
    }

    return nullptr;
}

sk_sp<sksg::RenderNode> CompositionBuilder::build(const AnimationBuilder& abuilder) {
    // First pass - transitively attach layer transform chains.
    for (auto& lbuilder : fLayerBuilders) {
        lbuilder.buildTransform(abuilder, this);
    }

    // Second pass - attach actual layer contents and finalize the layer render tree.
    std::vector<sk_sp<sksg::RenderNode>> layers;
    layers.reserve(fLayerBuilders.size());

    int prev_layer_index = -1;
    for (auto& lbuilder : fLayerBuilders) {
        if (auto layer = lbuilder.buildRenderTree(abuilder, this, prev_layer_index)) {
            layers.push_back(std::move(layer));
        }
        prev_layer_index = lbuilder.index();
    }

    if (layers.empty()) {
        return nullptr;
    }

    if (layers.size() == 1) {
        return std::move(layers[0]);
    }

    // Layers are painted in bottom->top order.
    std::reverse(layers.begin(), layers.end());
    layers.shrink_to_fit();

    return sksg::Group::Make(std::move(layers));
}

} // namespace internal
} // namespace skottie
