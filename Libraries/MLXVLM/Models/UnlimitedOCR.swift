//
//  UnlimitedOCR.swift
//  mlx-swift-lm
//
//  Unlimited-OCR delta over DeepSeek-OCR: native model_type registration and
//  Reference Sliding Window Attention (R-SWA) via RingSlidingKVCache.
//  Port of mlx_vlm.models.unlimited_ocr (config + unlimitedocr + processor).
//

import Foundation
import MLXLMCommon

/// Configuration for Unlimited-OCR HF packs (`model_type`: `unlimited-ocr` /
/// `unlimited_ocr`). Same JSON schema as DeepSeek-OCR plus R-SWA window fields.
public typealias UnlimitedOCRConfiguration = DeepseekOCRConfiguration

/// Processor configuration for Unlimited-OCR packs: the same
/// `preprocessor_config.json` schema as DeepSeek-OCR.
public typealias UnlimitedOCRProcessorConfiguration = DeepseekOCRProcessorConfiguration

extension DeepseekOCRProcessor {
    /// The Unlimited-OCR processor: the DeepSeek-OCR pipeline with Unlimited's two pack rules.
    ///
    /// One bare `<image>` whatever the page count (`prompt_utils.py:93`), and a 32-tile gundam
    /// cap (`processing_unlimitedocr.py:175`).
    public static func unlimited(
        _ config: UnlimitedOCRProcessorConfiguration, tokenizer: any Tokenizer
    ) -> DeepseekOCRProcessor {
        DeepseekOCRProcessor(
            config, tokenizer: tokenizer, imagePlaceholder: .single,
            defaultMaxNumTiles: unlimitedMaxNumTiles)
    }
}

/// Unlimited-OCR VLM: DeepSeek-OCR weights + R-SWA decode cache.
///
/// Python defaults `sliding_window_size` to 128 when unset; native Unlimited
/// routing always uses `RingSlidingKVCache` (never unbounded
/// `KVCacheSimple`).
///
/// ## Optional n-gram no-repeat
///
/// Upstream Unlimited-OCR / mlx-vlm leave the sliding-window no-repeat n-gram
/// logits processor **off** by default. Opt in by attaching the processor
/// through `GenerationComponents`:
///
/// ```swift
/// let parameters = GenerateParameters(temperature: 0, maxTokens: 8192)
/// let components = GenerationComponents(
///     logitProcessorFactory: { SlidingWindowNoRepeatNGramProcessor.unlimitedOCRSingleImage() }
/// )
/// // multi-page/PDF examples often use .unlimitedOCRMultiPage() (window 1024)
/// ```
///
/// Leaving `components` empty matches Python Unlimited defaults (guard off).
public final class UnlimitedOCR: DeepseekOCR {

    public override func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try makeLayerCaches(
            parameters: parameters, slidingWindowSize: config.resolvedSlidingWindowSize ?? 128)
    }
}
