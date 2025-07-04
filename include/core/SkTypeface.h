/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTypeface_DEFINED
#define SkTypeface_DEFINED

#include "include/core/SkFontArguments.h"
#include "include/core/SkFontParameters.h"
#include "include/core/SkFontStyle.h"
#include "include/core/SkFourByteTag.h"
#include "include/core/SkRect.h"
#include "include/core/SkRefCnt.h"
#include "include/core/SkSpan.h"
#include "include/core/SkString.h"
#include "include/core/SkTypes.h"
#include "include/private/SkWeakRefCnt.h"
#include "include/private/base/SkOnce.h"

#include <cstddef>
#include <cstdint>
#include <memory>

class SkData;
class SkDescriptor;
class SkFontMgr;
class SkFontDescriptor;
class SkScalerContext;
class SkStream;
class SkStreamAsset;
class SkWStream;
enum class SkTextEncoding;
struct SkAdvancedTypefaceMetrics;
struct SkScalerContextEffects;
struct SkScalerContextRec;

using SkTypefaceID = uint32_t;

/** Machine endian. */
typedef uint32_t SkFontTableTag;

/** \class SkTypeface

    The SkTypeface class specifies the typeface and intrinsic style of a font.
    This is used in the paint, along with optionally algorithmic settings like
    textSize, textSkewX, textScaleX, kFakeBoldText_Mask, to specify
    how text appears when drawn (and measured).

    Typeface objects are immutable, and so they can be shared between threads.
*/
class SK_API SkTypeface : public SkWeakRefCnt {
public:
    /** Returns the typeface's intrinsic style attributes. */
    SkFontStyle fontStyle() const;

    /** Returns true if style() has the kBold bit set. */
    bool isBold() const;

    /** Returns true if style() has the kItalic bit set. */
    bool isItalic() const;

    /** Returns true if the typeface claims to be fixed-pitch.
     *  This is a style bit, advance widths may vary even if this returns true.
     */
    bool isFixedPitch() const;

    /** Copy into 'coordinates' (allocated by the caller) the design variation coordinates.
     *
     *  @param coordinates the span into which to write the design variation coordinates.
     *
     *  @return The number of axes, or -1 if there is an error.
     *  If 'coordinates.size() >= numAxes' then 'coordinates' will be
     *  filled with the variation coordinates describing the position of this typeface in design
     *  variation space. It is possible the number of axes can be retrieved but actual position
     *  cannot.
     */
    int getVariationDesignPosition(
                       SkSpan<SkFontArguments::VariationPosition::Coordinate> coordinates) const;

    /** Copy into 'parameters' (allocated by the caller) the design variation parameters.
     *
     *  @param parameters the span into which to write the design variation parameters.
     *
     *  @return The number of axes, or -1 if there is an error.
     *  If 'parameters.size() >= numAxes' then 'parameters' will be
     *  filled with the variation parameters describing the position of this typeface in design
     *  variation space. It is possible the number of axes can be retrieved but actual parameters
     *  cannot.
     */
    int getVariationDesignParameters(SkSpan<SkFontParameters::Variation::Axis> parameters) const;

    /** Return a 32bit value for this typeface, unique for the underlying font
        data. Will never return 0.
     */
    SkTypefaceID uniqueID() const { return fUniqueID; }

    /** Returns true if the two typefaces reference the same underlying font,
        handling either being null (treating null as not equal to any font).
     */
    static bool Equal(const SkTypeface* facea, const SkTypeface* faceb);

    /** Returns a non-null typeface which contains no glyphs. */
    static sk_sp<SkTypeface> MakeEmpty();

    /** Return a new typeface based on this typeface but parameterized as specified in the
        SkFontArguments. If the SkFontArguments does not supply an argument for a parameter
        in the font then the value from this typeface will be used as the value for that
        argument. If the cloned typeface would be exaclty the same as this typeface then
        this typeface may be ref'ed and returned. May return nullptr on failure.
    */
    sk_sp<SkTypeface> makeClone(const SkFontArguments&) const;

    /**
     *  A typeface can serialize just a descriptor (names, etc.), or it can also include the
     *  actual font data (which can be large). This enum controls how serialize() decides what
     *  to serialize.
     */
    enum class SerializeBehavior {
        kDoIncludeData,
        kDontIncludeData,
        kIncludeDataIfLocal,
    };

    /** Write a unique signature to a stream, sufficient to reconstruct a
        typeface referencing the same font when Deserialize is called.
     */
    void serialize(SkWStream*, SerializeBehavior = SerializeBehavior::kIncludeDataIfLocal) const;

    /**
     *  Same as serialize(SkWStream*, ...) but returns the serialized data in SkData, instead of
     *  writing it to a stream.
     */
    sk_sp<SkData> serialize(SerializeBehavior = SerializeBehavior::kIncludeDataIfLocal) const;

    /** Given the data previously written by serialize(), return a new instance
        of a typeface referring to the same font. If that font is not available,
        return nullptr.
        Goes through all registered typeface factories and lastResortMgr (if non-null).
        Does not affect ownership of SkStream.
     */

    static sk_sp<SkTypeface> MakeDeserialize(SkStream*, sk_sp<SkFontMgr> lastResortMgr);

    /**
     *  Given an array of UTF32 character codes, return their corresponding glyph IDs.
     *
     *  @param unis span of UTF32 chars
     *  @param glyphs returns the corresponding glyph IDs for each character.
     */
    void unicharsToGlyphs(SkSpan<const SkUnichar> unis, SkSpan<SkGlyphID> glyphs) const;

    size_t textToGlyphs(const void* text, size_t byteLength, SkTextEncoding encoding,
                        SkSpan<SkGlyphID> glyphs) const;

    /**
     *  Return the glyphID that corresponds to the specified unicode code-point
     *  (in UTF32 encoding). If the unichar is not supported, returns 0.
     *
     *  This is a short-cut for calling unicharsToGlyphs().
     */
    SkGlyphID unicharToGlyph(SkUnichar unichar) const;

    /**
     *  Return the number of glyphs in the typeface.
     */
    int countGlyphs() const;

    // Table getters -- may fail if the underlying font format is not organized
    // as 4-byte tables.

    /** Return the number of tables in the font. */
    int countTables() const;

    /** Copy into tags[] (allocated by the caller) the list of table tags in
     *  the font, and return the number. This will be the same as CountTables()
     *  or 0 if an error occured. If tags is empty, this only returns the count
     *  (the same as calling countTables()).
     */
    int readTableTags(SkSpan<SkFontTableTag> tags) const;

    /** Given a table tag, return the size of its contents, or 0 if not present
     */
    size_t getTableSize(SkFontTableTag) const;

    /** Copy the contents of a table into data (allocated by the caller). Note
     *  that the contents of the table will be in their native endian order
     *  (which for most truetype tables is big endian). If the table tag is
     *  not found, or there is an error copying the data, then 0 is returned.
     *  If this happens, it is possible that some or all of the memory pointed
     *  to by data may have been written to, even though an error has occured.
     *
     *  @param tag  The table tag whose contents are to be copied
     *  @param offset The offset in bytes into the table's contents where the
     *  copy should start from.
     *  @param length The number of bytes, starting at offset, of table data
     *  to copy.
     *  @param data storage address where the table contents are copied to
     *  @return the number of bytes actually copied into data. If offset+length
     *  exceeds the table's size, then only the bytes up to the table's
     *  size are actually copied, and this is the value returned. If
     *  offset > the table's size, or tag is not a valid table,
     *  then 0 is returned.
     */
    size_t getTableData(SkFontTableTag tag, size_t offset, size_t length,
                        void* data) const;

    /**
     *  Return an immutable copy of the requested font table, or nullptr if that table was
     *  not found. This can sometimes be faster than calling getTableData() twice: once to find
     *  the length, and then again to copy the data.
     *
     *  @param tag  The table tag whose contents are to be copied
     *  @return an immutable copy of the table's data, or nullptr.
     */
    sk_sp<SkData> copyTableData(SkFontTableTag tag) const;

    /**
     *  Return the units-per-em value for this typeface, or zero if there is an
     *  error.
     */
    int getUnitsPerEm() const;

    /**
     *  Given a run of glyphs, return the associated horizontal adjustments.
     *  Adjustments are in "design units", which are integers relative to the
     *  typeface's units per em (see getUnitsPerEm).
     *
     *  Some typefaces are known to never support kerning. Calling this method
     *  with empty spans (e.g. getKerningPairAdustments({}, {})) returns
     *  a boolean indicating if the typeface might support kerning. If it
     *  returns false, then it will always return false (no kerning) for all
     *  possible glyph runs. If it returns true, then it *may* return true for
     *  some glyph runs.
     *
     *  If the method returns true, and there are 1 or more glyphs in the span, then
     *  this will return in adjustments N values,
     *  where N = min(glyphs.size() - 1, adjustments.size()).

     *  If the method returns false, then no kerning should be applied, and the adjustments
     *  array will be in an undefined state (possibly some values may have been
     *  written, but none of them should be interpreted as valid values).
     */
    bool getKerningPairAdjustments(SkSpan<const SkGlyphID> glyphs,
                                   SkSpan<int32_t> adjustments) const;

    struct LocalizedString {
        SkString fString;
        SkString fLanguage;
    };
    class LocalizedStrings {
    public:
        LocalizedStrings() = default;
        virtual ~LocalizedStrings() { }
        virtual bool next(LocalizedString* localizedString) = 0;
        void unref() { delete this; }

    private:
        LocalizedStrings(const LocalizedStrings&) = delete;
        LocalizedStrings& operator=(const LocalizedStrings&) = delete;
    };
    /**
     *  Returns an iterator which will attempt to enumerate all of the
     *  family names specified by the font.
     *  It is the caller's responsibility to unref() the returned pointer.
     */
    LocalizedStrings* createFamilyNameIterator() const;

    /**
     *  Return the family name for this typeface. It will always be returned
     *  encoded as UTF8, but the language of the name is whatever the host
     *  platform chooses.
     */
    void getFamilyName(SkString* name) const;

    /**
     *  Return the PostScript name for this typeface.
     *  Value may change based on variation parameters.
     *  Returns false if no PostScript name is available.
     */
    bool getPostScriptName(SkString* name) const;

    /**
     *  If the primary resource backing this typeface has a name (like a file
     *  path or URL) representable by unicode code points, the `resourceName`
     *  will be set. The primary purpose is as a user facing indication about
     *  where the data was obtained (which font file was used).
     *
     *  Returns the number of resources backing this typeface.
     *
     *  For local font collections resource name will often be a file path. The
     *  file path may or may not exist. If it does exist, using it to create an
     *  SkTypeface may or may not create a similar SkTypeface to this one.
     */
    int getResourceName(SkString* resourceName) const;

    /**
     *  Return a stream for the contents of the font data, or NULL on failure.
     *  If ttcIndex is not null, it is set to the TrueTypeCollection index
     *  of this typeface within the stream, or 0 if the stream is not a
     *  collection.
     *  The caller is responsible for deleting the stream.
     */
    std::unique_ptr<SkStreamAsset> openStream(int* ttcIndex) const;

    /**
     * Return a stream for the contents of the font data.
     * Returns nullptr on failure or if the font data isn't already available in stream form.
     * Use when the stream can be used opportunistically but the calling code would prefer
     * to fall back to table access if creating the stream would be expensive.
     * Otherwise acts the same as openStream.
     */
    std::unique_ptr<SkStreamAsset> openExistingStream(int* ttcIndex) const;

    /**
     *  Return a scalercontext for the given descriptor. It may return a
     *  stub scalercontext that will not crash, but will draw nothing.
     */
    std::unique_ptr<SkScalerContext> createScalerContext(const SkScalerContextEffects&,
                                                         const SkDescriptor*) const;

    /**
     *  Return a rectangle (scaled to 1-pt) that represents the union of the bounds of all
     *  of the glyphs, but each one positioned at (0,). This may be conservatively large, and
     *  will not take into account any hinting or other size-specific adjustments.
     */
    SkRect getBounds() const;

    // PRIVATE / EXPERIMENTAL -- do not call
    void filterRec(SkScalerContextRec* rec) const {
        this->onFilterRec(rec);
    }
    // PRIVATE / EXPERIMENTAL -- do not call
    void getFontDescriptor(SkFontDescriptor* desc, bool* isLocal) const {
        this->onGetFontDescriptor(desc, isLocal);
    }
    // PRIVATE / EXPERIMENTAL -- do not call
    void* internal_private_getCTFontRef() const {
        return this->onGetCTFontRef();
    }

    /* Skia reserves all tags that begin with a lower case letter and 0 */
    using FactoryId = SkFourByteTag;
    static void Register(
            FactoryId id,
            sk_sp<SkTypeface> (*make)(std::unique_ptr<SkStreamAsset>, const SkFontArguments&));

#ifdef SK_SUPPORT_UNSPANNED_APIS
public:
    int getVariationDesignPosition(SkFontArguments::VariationPosition::Coordinate coordinates[],
                                   int count) const {
        return this->getVariationDesignPosition({coordinates, count});
    }
    int getVariationDesignParameters(SkFontParameters::Variation::Axis parameters[],
                                     int count) const {
        return this->getVariationDesignParameters({parameters, count});
    }
    void unicharsToGlyphs(const SkUnichar unis[], int count, SkGlyphID glyphs[]) const {
        this->unicharsToGlyphs({unis, count}, {glyphs, count});
    }
    int textToGlyphs(const void* text, size_t byteLength, SkTextEncoding encoding,
                     SkGlyphID glyphs[], int maxGlyphCount) const {
        return (int)this->textToGlyphs(text, byteLength, encoding, {glyphs, maxGlyphCount});
    }
    int getTableTags(SkFontTableTag tags[]) const {
        const size_t count = tags ? MAX_REASONABLE_TABLE_COUNT : 0;
        return this->readTableTags({tags, count});
    }
    bool getKerningPairAdjustments(const SkGlyphID glyphs[], int count,
                                   int32_t adjustments[]) const {
        return this->getKerningPairAdjustments({glyphs, count}, {adjustments, count});
    }
#endif

protected:
    // needed until onGetTableTags() is updated to take a span
    enum { MAX_REASONABLE_TABLE_COUNT = (1 << 16) - 1 };

    explicit SkTypeface(const SkFontStyle& style, bool isFixedPitch = false);
    ~SkTypeface() override;

    virtual sk_sp<SkTypeface> onMakeClone(const SkFontArguments&) const = 0;

    /** Sets the fixedPitch bit. If used, must be called in the constructor. */
    void setIsFixedPitch(bool isFixedPitch) { fIsFixedPitch = isFixedPitch; }
    /** Sets the font style. If used, must be called in the constructor. */
    void setFontStyle(SkFontStyle style) { fStyle = style; }

    virtual SkFontStyle onGetFontStyle() const; // TODO: = 0;

    virtual bool onGetFixedPitch() const; // TODO: = 0;

    // Must return a valid scaler context. It can not return nullptr.
    virtual std::unique_ptr<SkScalerContext> onCreateScalerContext(
        const SkScalerContextEffects&, const SkDescriptor*) const = 0;
    virtual std::unique_ptr<SkScalerContext> onCreateScalerContextAsProxyTypeface
        (const SkScalerContextEffects&, const SkDescriptor*, SkTypeface* proxyTypeface) const;
    virtual void onFilterRec(SkScalerContextRec*) const = 0;
    friend class SkScalerContext;  // onFilterRec

    //  Subclasses *must* override this method to work with the PDF backend.
    virtual std::unique_ptr<SkAdvancedTypefaceMetrics> onGetAdvancedMetrics() const = 0;
    // For type1 postscript fonts only, set the glyph names for each glyph.
    // destination array is non-null, and points to an array of size this->countGlyphs().
    // Backends that do not suport type1 fonts should not override.
    virtual void getPostScriptGlyphNames(SkString*) const = 0;

    // The mapping from glyph to Unicode; array indices are glyph ids.
    // For each glyph, give the default Unicode value, if it exists.
    // dstArray is non-null, and points to an array of size this->countGlyphs().
    virtual void getGlyphToUnicodeMap(SkSpan<SkUnichar> dstArray) const = 0;

    virtual std::unique_ptr<SkStreamAsset> onOpenStream(int* ttcIndex) const = 0;

    virtual std::unique_ptr<SkStreamAsset> onOpenExistingStream(int* ttcIndex) const;

    virtual bool onGlyphMaskNeedsCurrentColor() const = 0;

    virtual int onGetVariationDesignPosition(
                                 SkSpan<SkFontArguments::VariationPosition::Coordinate>) const = 0;

    virtual int onGetVariationDesignParameters(SkSpan<SkFontParameters::Variation::Axis>) const = 0;

    virtual void onGetFontDescriptor(SkFontDescriptor*, bool* isLocal) const = 0;

    virtual void onCharsToGlyphs(SkSpan<const SkUnichar>, SkSpan<SkGlyphID>) const = 0;
    virtual int onCountGlyphs() const = 0;

    virtual int onGetUPEM() const = 0;
    virtual bool onGetKerningPairAdjustments(SkSpan<const SkGlyphID>,
                                             SkSpan<int32_t> adjustments) const;

    /** Returns the family name of the typeface as known by its font manager.
     *  This name may or may not be produced by the family name iterator.
     */
    virtual void onGetFamilyName(SkString* familyName) const = 0;
    virtual bool onGetPostScriptName(SkString*) const = 0;
    virtual int onGetResourceName(SkString* resourceName) const; // TODO: = 0;

    /** Returns an iterator over the family names in the font. */
    virtual LocalizedStrings* onCreateFamilyNameIterator() const = 0;

    virtual int onGetTableTags(SkSpan<SkFontTableTag>) const = 0;
    virtual size_t onGetTableData(SkFontTableTag, size_t offset,
                                  size_t length, void* data) const = 0;
    virtual sk_sp<SkData> onCopyTableData(SkFontTableTag) const;

    virtual bool onComputeBounds(SkRect*) const;

    virtual void* onGetCTFontRef() const { return nullptr; }

private:
    /** Returns true if the typeface's glyph masks may refer to the foreground
     *  paint foreground color. This is needed to determine caching requirements. Usually true for
     *  typefaces that contain a COLR table.
     */
    bool glyphMaskNeedsCurrentColor() const;
    friend class SkStrikeServerImpl;  // glyphMaskNeedsCurrentColor
    friend class SkTypefaceProxyPrototype;  // glyphMaskNeedsCurrentColor

    /** Retrieve detailed typeface metrics.  Used by the PDF backend.  */
    std::unique_ptr<SkAdvancedTypefaceMetrics> getAdvancedMetrics() const;
    friend class SkRandomTypeface;   // getAdvancedMetrics
    friend class SkPDFFont;          // getAdvancedMetrics
    friend class SkTypeface_proxy;
    friend class SkFontPriv;         // getGlyphToUnicodeMap
    friend void TestSkTypefaceGlyphToUnicodeMap(SkTypeface&, SkSpan<SkUnichar>);

private:
    SkTypefaceID        fUniqueID;
    SkFontStyle         fStyle;
    mutable SkRect      fBounds;
    mutable SkOnce      fBoundsOnce;
    bool                fIsFixedPitch;

    using INHERITED = SkWeakRefCnt;
};
#endif
