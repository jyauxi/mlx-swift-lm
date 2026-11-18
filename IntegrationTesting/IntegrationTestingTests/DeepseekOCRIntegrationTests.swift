// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - DeepSeek-OCR (deepseekocr) IntegrationTesting example
//
// Proves first-class DeepSeek-OCR via `VLMRegistry.deepseekOCR5bit`
// (`model_type=deepseekocr`) — not Unlimited remap, `UnlimitedOCR`, or R-SWA.
//
// Cache-gated on `mlx-community/DeepSeek-OCR-5bit`, or download it with
// `TEST_RUNNER_MLX_RUN_DEEPSEEK_OCR_INTEGRATION=1` (xcodebuild drops the prefix). OCR content quality is
// not asserted — load + prepare + short generate must succeed on the DeepSeek path.

private let deepseekOCRModelId = "mlx-community/DeepSeek-OCR-5bit"
private let unlimitedOCRModelId = "mlx-community/Unlimited-OCR-6bit"

private var shouldRunDeepseekOCRIntegration: Bool {
    ProcessInfo.processInfo.environment["MLX_RUN_DEEPSEEK_OCR_INTEGRATION"] == "1"
        || hfSnapshotDir(modelId: deepseekOCRModelId) != nil
}

extension ModelLoadingSuite {
    @Suite(
        .serialized,
        .timeLimit(.minutes(15)),
        .enabled(if: shouldRunDeepseekOCRIntegration))
    struct DeepseekOCRIntegrationTests {}
}

extension ModelLoadingSuite.DeepseekOCRIntegrationTests {

    @Test
    func loadsDeepseekOCRAndRunsOCR() async throws {
        let container = try await TestModelManager.shared.load("deepseekocr-container") {
            // Keep Hub `_orig_model_type` from diverting loads to UnlimitedOCR.
            let tokenizerLoader = #huggingFaceTokenizerLoader()
            let deepseekFactory = VLMModelFactory(
                typeRegistry: VLMTypeRegistry.shared,
                processorRegistry: VLMProcessorTypeRegistry.shared,
                modelRegistry: VLMRegistry.shared,
                honorOrigModelType: false)
            if let dir = hfSnapshotDir(modelId: deepseekOCRModelId) {
                return try await deepseekFactory.loadContainer(
                    from: dir, using: tokenizerLoader)
            }
            return try await deepseekFactory.loadContainer(
                from: #hubDownloader(),
                using: tokenizerLoader,
                configuration: VLMRegistry.deepseekOCR5bit)
        }

        let isDeepseek = await container.perform { $0.model is DeepseekOCR }
        let isUnlimited = await container.perform { $0.model is UnlimitedOCR }
        #expect(isDeepseek, "expected DeepseekOCR for deepseekocr path")
        #expect(!isUnlimited, "must not load UnlimitedOCR / R-SWA here")

        // Synthetic page larger than one 640 tile, so gundam crops local tiles; asserts
        // path health, not OCR accuracy.
        let page = VisionTestImages.solidColor(.white, size: 1280)
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 32, temperature: 0),
            // Keep native resolution (ChatSession default 512² best-fit breaks tiling).
            processing: .init())

        let text = try await session.respond(
            to: "Free OCR.",
            image: .ciImage(page))
        #expect(!text.isEmpty, "DeepSeek-OCR produced empty output")
    }
}

// MARK: - Unlimited-OCR chat template
//
// The processor the `UnlimitedOCRProcessor` key builds must render one `<image>`
// through the shipped pack's own chat template. Needs the pack's tokenizer and
// processor files, not its weights.

@Suite(
    .serialized,
    .enabled(if: hfSnapshotDir(modelId: unlimitedOCRModelId) != nil))
struct UnlimitedOCRTokenizerIntegrationTests {

    @Test
    func checkpointTemplateRendersImagePlaceholder() async throws {
        let directory = try #require(hfSnapshotDir(modelId: unlimitedOCRModelId))
        let tokenizer = TemplateRecordingTokenizer(
            try await #huggingFaceTokenizerLoader().load(from: directory))
        let configURL = try #require(
            ["preprocessor_config.json", "processor_config.json"]
                .map { directory.appending(component: $0) }
                .first { FileManager.default.fileExists(atPath: $0.path) })
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: try Data(contentsOf: configURL),
            processorType: "UnlimitedOCRProcessor",
            tokenizer: tokenizer)

        _ = try await processor.prepare(
            input: UserInput(
                prompt: "document parsing.",
                images: [.ciImage(VisionTestImages.solidColor(.white, size: 64))]))

        let render = try #require(tokenizer.renders.first)
        let imageToken = try #require(tokenizer.convertTokenToId("<image>"))
        #expect(render.contents == ["<image>document parsing."])
        #expect(render.tokens.count(where: { $0 == imageToken }) == 1)
    }
}

/// Forwards to a real tokenizer and records each chat-template render.
private final class TemplateRecordingTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let inner: any MLXLMCommon.Tokenizer
    private let lock = NSLock()
    private var _renders = [(contents: [String], tokens: [Int])]()

    init(_ inner: any MLXLMCommon.Tokenizer) { self.inner = inner }

    var renders: [(contents: [String], tokens: [Int])] { lock.withLock { _renders } }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { inner.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { inner.convertIdToToken(id) }
    var bosToken: String? { inner.bosToken }
    var eosToken: String? { inner.eosToken }
    var unknownToken: String? { inner.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let tokens = try inner.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
        lock.withLock {
            _renders.append((messages.compactMap { $0["content"] as? String }, tokens))
        }
        return tokens
    }
}
