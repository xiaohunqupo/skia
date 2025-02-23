/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "include/core/SkTypes.h"

#if defined(SK_SUPPORT_PDF)
#include "include/core/SkData.h"
#include "include/core/SkDocument.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkStream.h"
#include "include/core/SkString.h"
#include "include/docs/SkPDFDocument.h"
#include "include/docs/SkPDFJpegHelpers.h"
#include "src/pdf/SkPDFUtils.h"
#include "tests/Test.h"

#include <cstdint>
#include <cstring>

DEF_TEST(SkPDF_Metadata, r) {
    REQUIRE_PDF_DOCUMENT(SkPDF_Metadata, r);
    SkPDF::DateTime now;
    SkPDFUtils::GetDateTime(&now);
    SkPDF::Metadata metadata;
    metadata.fTitle = "A1";
    metadata.fAuthor = "A2";
    metadata.fSubject = "A3";
    metadata.fKeywords = "A4";
    metadata.fCreator = "A5";
    metadata.fCreation = now;
    metadata.fModified = now;
    metadata.jpegDecoder = SkPDF::JPEG::Decode;
    metadata.jpegEncoder = SkPDF::JPEG::Encode;

    SkDynamicMemoryWStream pdf;
    auto doc = SkPDF::MakeDocument(&pdf, metadata);
    doc->beginPage(612.0f, 792.0f);
    doc->close();
    sk_sp<SkData> data = pdf.detachAsData();
    static const char* expectations[] = {
        "/Title (A1)",
        "/Author (A2)",
        "/Subject (A3)",
        "/Keywords (A4)",
        "/Creator (A5)",
        "/Producer (Skia/PDF ",
        "/CreationDate (D:",
        "/ModDate (D:"
    };
    const uint8_t* bytes = data->bytes();
    for (const char* expectation : expectations) {
        size_t len = strlen(expectation);
        bool found = false;
        size_t N = 1 + data->size() - len;
        for (size_t i = 0; i < N; ++i) {
            if (0 == memcmp(bytes + i, expectation, len)) {
                found = true;
                break;
            }
        }
        if (!found) {
            ERRORF(r, "expectation missing: '%s'.", expectation);
        }
    }
}
#endif
