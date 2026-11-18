// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class VLMRegistryTests: XCTestCase {

    func testGemma4VLMRegistryUsesTurnEndToken() {
        for configuration in [
            VLMRegistry.gemma4_E2B_it_4bit,
            VLMRegistry.gemma4_E4B_it_4bit,
            VLMRegistry.gemma4_31B_it_4bit,
            VLMRegistry.gemma4_26BA4B_it_4bit,
        ] {
            XCTAssertEqual(configuration.extraEOSTokens, ["<turn|>"])
        }
    }

    /// The pack's `chat_template.jinja` appends its own separator after the
    /// rendered content (`{{message['content']}} `), so a `defaultPrompt`
    /// with a trailing (or leading) space would double up whitespace in the
    /// tokenized prompt. Every DeepSeek-OCR / Unlimited-OCR registry entry's
    /// `defaultPrompt` must be pre-trimmed.
    func testOCRRegistryDefaultPromptsHaveNoSurroundingWhitespace() {
        for configuration in [VLMRegistry.deepseekOCR5bit, VLMRegistry.unlimitedOCR6bit] {
            let prompt = configuration.defaultPrompt
            XCTAssertEqual(
                prompt, prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                "\(configuration.name) defaultPrompt has leading/trailing whitespace: \(String(reflecting: prompt))"
            )
        }
    }

    func testDeepseekOCRModelTypeLoadsPinnedHFConfig() async throws {
        let contains = await VLMTypeRegistry.shared.contains("deepseekocr")
        XCTAssertTrue(contains)
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Self.deepseekOCRConfig.data(using: .utf8)!,
            modelType: "deepseekocr")
        XCTAssertTrue(model is DeepseekOCR)
        XCTAssertEqual(VLMRegistry.deepseekOCR5bit.name, "mlx-community/DeepSeek-OCR-5bit")
    }

    func testDeepseekOCRProcessorAliasesLoadPinnedHFConfig() async throws {
        for processorType in ["DeepseekOCRProcessor", "DeepseekVLV2Processor"] {
            let processor = try await VLMProcessorTypeRegistry.shared.createModel(
                configuration: Self.deepseekOCRProcessorConfig.data(using: .utf8)!,
                processorType: processorType,
                tokenizer: StubTokenizer())
            XCTAssertTrue(processor is DeepseekOCRProcessor)
        }
        // The placeholder the lattice is masked against. Construction succeeds
        // whatever this says, so assert it: a fixture drifting off the shipped
        // packs' "<image>" would mis-mask every image and no test would notice.
        let decoded = try JSONDecoder().decode(
            DeepseekOCRProcessorConfiguration.self,
            from: Self.deepseekOCRProcessorConfig.data(using: .utf8)!)
        XCTAssertEqual(decoded.imageToken, "<image>")
    }

    func testDeepseekOCRFactoryFailsGracefullyWhenWeightsAreMissing() async throws {
        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try Self.deepseekOCRConfig.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("config.json"))
        try Self.deepseekOCRProcessorConfig.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("processor_config.json"))

        do {
            _ = try await VLMModelFactory.shared.load(from: directory, using: StubTokenizerLoader())
            XCTFail("expected missing weights to throw")
        } catch {
            XCTAssertFalse((error as NSError).localizedDescription.isEmpty)
        }
    }

    func testUnlimitedOCRModelTypesLoadPinnedHFConfig() async throws {
        for modelType in ["unlimited-ocr", "unlimited_ocr"] {
            let contains = await VLMTypeRegistry.shared.contains(modelType)
            XCTAssertTrue(contains, modelType)
            let model = try await VLMTypeRegistry.shared.createModel(
                configuration: Self.unlimitedOCRConfig.data(using: .utf8)!,
                modelType: modelType)
            XCTAssertTrue(model is UnlimitedOCR, modelType)
            let unlimited = try XCTUnwrap(model as? UnlimitedOCR)
            XCTAssertEqual(unlimited.config.resolvedSlidingWindowSize, 128)
            let caches = try unlimited.newCache(parameters: nil)
            XCTAssertEqual(caches.count, 12)
            for cache in caches {
                let ring = try XCTUnwrap(cache as? RingSlidingKVCache)
                XCTAssertEqual(ring.windowSize, 128)
            }
        }
        XCTAssertEqual(VLMRegistry.unlimitedOCR6bit.name, "mlx-community/Unlimited-OCR-6bit")
    }

    /// Unlimited-OCR-2 keeps `model_type: unlimited-ocr` but adds sparse attention, which
    /// this port does not build, so its config must fail to load instead of running as v1.
    func testUnlimitedOCR2ConfigIsRefused() async throws {
        let sparseKeys: [String: Any] = [
            "use_sparse_attention": true, "sparse_attention_config": [String: Any](),
        ]
        for (key, value) in sparseKeys {
            var json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(Self.unlimitedOCRConfig.utf8))
                    as? [String: Any])
            json[key] = value
            let data = try JSONSerialization.data(withJSONObject: json)
            for modelType in ["unlimited-ocr", "unlimited_ocr"] {
                do {
                    _ = try await VLMTypeRegistry.shared.createModel(
                        configuration: data, modelType: modelType)
                    XCTFail("\(modelType) with \(key) must not load")
                } catch {
                    XCTAssertTrue(
                        "\(error)".contains("Unlimited-OCR-2 is not supported"), "\(error)")
                }
            }
        }

        // The shipped v1 and DeepSeek configs still decode.
        for config in [Self.unlimitedOCRConfig, Self.deepseekOCRConfig] {
            XCTAssertNoThrow(
                try JSONDecoder().decode(DeepseekOCRConfiguration.self, from: Data(config.utf8)))
        }
    }

    /// A section that is present but null is not a sparse-attention config.
    func testNullTextSectionStillDecodes() throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(Self.unlimitedOCRConfig.utf8))
                as? [String: Any])
        json["text_config"] = NSNull()
        let data = try JSONSerialization.data(withJSONObject: json)
        XCTAssertNoThrow(try JSONDecoder().decode(DeepseekOCRConfiguration.self, from: data))
    }

    func testUnlimitedOCRProcessorKeyLoadsPinnedHFConfig() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Self.unlimitedOCRProcessorConfig.data(using: .utf8)!,
            processorType: "UnlimitedOCRProcessor",
            tokenizer: StubTokenizer())
        XCTAssertTrue(processor is DeepseekOCRProcessor)
        let decoded = try JSONDecoder().decode(
            UnlimitedOCRProcessorConfiguration.self,
            from: Self.unlimitedOCRProcessorConfig.data(using: .utf8)!)
        XCTAssertEqual(decoded.imageToken, "<image>")
    }

    /// The shipped Unlimited pack declares `model_type: deepseekocr`, so nothing but the
    /// `_orig_model_type` shim keeps it off the plain DeepSeek path — where it would load,
    /// generate, and grow an unbounded cache instead of the ring. This pins that routing
    /// through the production registry at the production default, so flipping
    /// `defaultHonorOrigModelType` fails here rather than in someone's memory graph.
    func testShippedUnlimitedPackRoutesToRingCacheUnderTheProductionDefault() async throws {
        let data = Self.unlimitedOCRConfig.data(using: .utf8)!
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        XCTAssertEqual(base.modelType, "deepseekocr")
        XCTAssertEqual(base.origModelType, "unlimited-ocr")

        let resolved = base.resolvedModelType(
            honorOrigModelType: VLMModelFactory.defaultHonorOrigModelType)
        XCTAssertEqual(
            resolved, "unlimited-ocr",
            "the shipped pack must resolve to Unlimited under the shipped default")

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: resolved)
        let unlimited = try XCTUnwrap(
            model as? UnlimitedOCR, "resolved \(resolved) built \(type(of: model))")
        for cache in try unlimited.newCache(parameters: nil) {
            let ring = try XCTUnwrap(cache as? RingSlidingKVCache)
            XCTAssertEqual(ring.windowSize, 128)
        }
    }

    /// The test above resolves the type by hand and asks the registry directly; this one
    /// hands the shipped pack's `config.json` to `VLMModelFactory` under the shared
    /// factory's policy and records which registry key the factory's own resolution
    /// invoked — then checks that the production registry builds `UnlimitedOCR` for it.
    /// A factory that stopped honoring `_orig_model_type` would select `deepseekocr`
    /// here and nothing else in this file would notice.
    func testShippedUnlimitedPackSelectsTheUnlimitedCreatorThroughTheFactory() async throws {
        let data = Self.unlimitedOCRConfig.data(using: .utf8)!
        let factory = Self.makeProbeFactory(
            registeredTypes: ["deepseekocr", "unlimited-ocr", "unlimited_ocr"],
            honorOrigModelType: VLMModelFactory.shared.honorOrigModelType)

        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appendingPathComponent("config.json"))

        let selected = try await Self.selectedModelType(factory: factory, directory: directory)
        XCTAssertEqual(selected, "unlimited-ocr")

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: selected)
        XCTAssertTrue(model is UnlimitedOCR, "\(selected) built \(type(of: model))")
    }

    /// The registered `mlx-community/Unlimited-OCR-6bit` pack ships the native type.
    func testNativeUnlimitedPackSelectsTheUnlimitedCreatorThroughTheFactory() async throws {
        var json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(Self.unlimitedOCRConfig.utf8)) as? [String: Any]
        )
        json["model_type"] = "unlimited-ocr"
        json["_orig_model_type"] = nil
        let data = try JSONSerialization.data(withJSONObject: json)
        let factory = Self.makeProbeFactory(
            registeredTypes: ["deepseekocr", "unlimited-ocr", "unlimited_ocr"],
            honorOrigModelType: false)

        let directory = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appendingPathComponent("config.json"))

        let selected = try await Self.selectedModelType(factory: factory, directory: directory)
        XCTAssertEqual(selected, "unlimited-ocr")
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: selected)
        XCTAssertTrue(model is UnlimitedOCR, "\(selected) built \(type(of: model))")
    }

    /// The same pack with the policy off is the regression this guards against: it builds
    /// plain DeepSeek-OCR, whose caches grow without bound.
    func testShippedUnlimitedPackFallsToUnboundedCacheWhenThePolicyIsOff() async throws {
        let data = Self.unlimitedOCRConfig.data(using: .utf8)!
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let resolved = base.resolvedModelType(honorOrigModelType: false)
        XCTAssertEqual(resolved, "deepseekocr")

        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: data, modelType: resolved)
        XCTAssertTrue(model is DeepseekOCR)
        XCTAssertFalse(model is UnlimitedOCR)
    }

    // MARK: - `_orig_model_type` policy
    //
    // Each creator in the private registry throws `CreatorProbe` naming its
    // registry key, so a load proves which key the factory selected without
    // building a model or needing weights.

    /// Most `VLMTypeRegistry` keys are underscore-spelled (`qwen3_vl`, `glm_ocr`, …),
    /// so the shim must reach an underscore key as spelled, case-insensitively.
    func testKnownOrigModelTypeSelectsItsCreatorOverModelType() async throws {
        let factory = Self.makeProbeFactory(
            registeredTypes: ["base-arch", "native_arch"], honorOrigModelType: true)
        let selected = try await Self.selectedModelType(
            factory: factory, modelType: "base-arch", origModelType: "Native_Arch")
        XCTAssertEqual(selected, "native_arch")
    }

    func testUnknownOrigModelTypeFallsBackToModelTypeCreator() async throws {
        let factory = Self.makeProbeFactory(
            registeredTypes: ["base-arch"], honorOrigModelType: true)
        let selected = try await Self.selectedModelType(
            factory: factory, modelType: "base-arch", origModelType: "unregistered-arch")
        XCTAssertEqual(selected, "base-arch")
    }

    func testDisabledPolicyIgnoresKnownOrigModelType() async throws {
        let factory = Self.makeProbeFactory(
            registeredTypes: ["base-arch", "native-arch"], honorOrigModelType: false)
        let selected = try await Self.selectedModelType(
            factory: factory, modelType: "base-arch", origModelType: "native-arch")
        XCTAssertEqual(selected, "base-arch")
    }

    func testOrigModelTypeCreatorFailureOtherThanRegistryMissPropagates() async throws {
        // A creator registered for the original type that itself reports a
        // *different* unsupported type is not a registry miss on the original
        // type, so the factory must surface it rather than fall back.
        let registry = ModelTypeRegistry<LanguageModel>(creators: [
            "base-arch": { _ in throw CreatorProbe(modelType: "base-arch") },
            "native-arch": { _ in throw ModelFactoryError.unsupportedModelType("inner-arch") },
        ])
        let factory = VLMModelFactory(
            typeRegistry: registry, processorRegistry: VLMProcessorTypeRegistry.shared,
            modelRegistry: VLMRegistry.shared, honorOrigModelType: true)
        let directory = try Self.writeConfig(modelType: "base-arch", origModelType: "native-arch")
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await factory.load(from: directory, using: StubTokenizerLoader())
            XCTFail("expected the creator's error to propagate")
        } catch ModelFactoryError.unsupportedModelType(let type) {
            XCTAssertEqual(type, "inner-arch")
        }
    }

    func testNeitherOrigNorModelTypeRegisteredReportsModelType() async throws {
        let factory = Self.makeProbeFactory(registeredTypes: [], honorOrigModelType: true)
        let directory = try Self.writeConfig(modelType: "base-arch", origModelType: "native-arch")
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await factory.load(from: directory, using: StubTokenizerLoader())
            XCTFail("expected an unsupported model type error")
        } catch ModelFactoryError.unsupportedModelType(let type) {
            XCTAssertEqual(type, "base-arch")
        }
    }

    private struct CreatorProbe: Error {
        let modelType: String
    }

    private static func makeProbeFactory(
        registeredTypes: [String], honorOrigModelType: Bool
    ) -> VLMModelFactory {
        var creators: [String: (Data) throws -> LanguageModel] = [:]
        for type in registeredTypes {
            creators[type] = { _ in throw CreatorProbe(modelType: type) }
        }
        return VLMModelFactory(
            typeRegistry: ModelTypeRegistry(creators: creators),
            processorRegistry: VLMProcessorTypeRegistry.shared,
            modelRegistry: VLMRegistry.shared,
            honorOrigModelType: honorOrigModelType)
    }

    /// Loads a config with the given types through `factory` and returns the
    /// registry key whose creator the factory invoked.
    private static func selectedModelType(
        factory: VLMModelFactory, modelType: String, origModelType: String
    ) async throws -> String {
        let directory = try writeConfig(modelType: modelType, origModelType: origModelType)
        defer { try? FileManager.default.removeItem(at: directory) }
        return try await selectedModelType(factory: factory, directory: directory)
    }

    /// Loads the pack at `directory` through `factory` and returns the registry key
    /// whose creator the factory invoked.
    private static func selectedModelType(
        factory: VLMModelFactory, directory: URL
    ) async throws -> String {
        do {
            _ = try await factory.load(from: directory, using: StubTokenizerLoader())
        } catch let probe as CreatorProbe {
            return probe.modelType
        }
        XCTFail("load finished without invoking a probe creator")
        return ""
    }

    private static func writeConfig(modelType: String, origModelType: String) throws -> URL {
        let directory = try makeTemporaryDirectory()
        let json = """
            {"model_type": "\(modelType)", "_orig_model_type": "\(origModelType)"}
            """
        try json.data(using: .utf8)!.write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VLMRegistryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

private final class StubTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any Tokenizer {
        StubTokenizer()
    }
}

private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

extension VLMRegistryTests {
    fileprivate static let deepseekOCRConfig = #"""
        {
         "architectures": [
          "DeepseekOCRForCausalLM"
         ],
         "auto_map": {
          "AutoConfig": "modeling_deepseekocr.DeepseekOCRConfig",
          "AutoModel": "modeling_deepseekocr.DeepseekOCRForCausalLM"
         },
         "bos_token_id": 0,
         "candidate_resolutions": [[1024, 1024]],
         "eos_token_id": 1,
         "first_k_dense_replace": 1,
         "global_view_pos": "head",
         "hidden_size": 1280,
         "intermediate_size": 6848,
         "kv_lora_rank": null,
         "language_config": {
          "architectures": ["DeepseekV2ForCausalLM"],
          "bos_token_id": 0,
          "eos_token_id": 1,
          "first_k_dense_replace": 1,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "kv_lora_rank": null,
          "lm_head": true,
          "max_position_embeddings": 8192,
          "moe_intermediate_size": 896,
          "n_group": 1,
          "n_routed_experts": 64,
          "n_shared_experts": 2,
          "num_attention_heads": 10,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 12,
          "num_key_value_heads": 10,
          "q_lora_rank": null,
          "qk_nope_head_dim": 0,
          "qk_rope_head_dim": 0,
          "rm_head": false,
          "topk_group": 1,
          "topk_method": "greedy",
          "torch_dtype": "bfloat16",
          "use_mla": false,
          "v_head_dim": 0,
          "vocab_size": 129280
         },
         "lm_head": true,
         "max_position_embeddings": 8192,
         "model_type": "deepseekocr",
         "moe_intermediate_size": 896,
         "n_group": 1,
         "n_routed_experts": 64,
         "n_shared_experts": 2,
         "num_attention_heads": 10,
         "num_experts_per_tok": 6,
         "num_hidden_layers": 12,
         "num_key_value_heads": 10,
         "projector_config": {
          "input_dim": 2048,
          "model_type": "mlp_projector",
          "n_embed": 1280,
          "projector_type": "linear"
         },
         "q_lora_rank": null,
         "qk_nope_head_dim": 0,
         "qk_rope_head_dim": 0,
         "quantization": {
          "group_size": 64,
          "bits": 5,
          "mode": "affine"
         },
         "quantization_config": {
          "group_size": 64,
          "bits": 5,
          "mode": "affine"
         },
         "rm_head": false,
         "tile_tag": "2D",
         "topk_group": 1,
         "topk_method": "greedy",
         "transformers_version": "4.46.3",
         "use_mla": false,
         "v_head_dim": 0,
         "vision_config": {
          "image_size": 1024,
          "mlp_ratio": 3.7362,
          "model_name": "deeplip_b_l",
          "model_type": "vision",
          "width": {
           "clip-l-14-224": {
            "heads": 16,
            "image_size": 224,
            "layers": 24,
            "patch_size": 14,
            "width": 1024
           },
           "sam_vit_b": {
            "downsample_channels": [512, 1024],
            "global_attn_indexes": [2, 5, 8, 11],
            "heads": 12,
            "layers": 12,
            "width": 768
           }
          }
         },
         "vocab_size": 129280
        }
        """#

    fileprivate static let deepseekOCRProcessorConfig = #"""
        {
         "add_special_token": false,
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "ignore_id": -100,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": "<image>",
         "mask_prompt": false,
         "normalize": true,
         "pad_token": "<｜▁pad▁｜>",
         "patch_size": 16,
         "processor_class": "DeepseekVLV2Processor",
         "sft_format": "deepseek"
        }
        """#

    /// The `majentik/Unlimited-OCR-MLX-6bit` shape, verbatim in the fields that decide
    /// routing: `model_type: deepseekocr` with `_orig_model_type: unlimited-ocr`.
    /// `mlx-community/Unlimited-OCR-6bit` ships `model_type: unlimited-ocr` instead.
    fileprivate static let unlimitedOCRConfig = #"""
        {
         "architectures": [
          "UnlimitedOCRForCausalLM"
         ],
         "bos_token_id": 0,
         "candidate_resolutions": [[1024, 1024]],
         "eos_token_id": 1,
         "first_k_dense_replace": 1,
         "language_config": {
          "architectures": ["DeepseekV2ForCausalLM"],
          "bos_token_id": 0,
          "eos_token_id": 1,
          "first_k_dense_replace": 1,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "lm_head": true,
          "max_position_embeddings": 8192,
          "moe_intermediate_size": 896,
          "n_group": 1,
          "n_routed_experts": 64,
          "n_shared_experts": 2,
          "num_attention_heads": 10,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 12,
          "num_key_value_heads": 10,
          "sliding_window_size": 128,
          "topk_group": 1,
          "topk_method": "greedy",
          "vocab_size": 129280
         },
         "lm_head": true,
         "max_position_embeddings": 8192,
         "model_type": "deepseekocr",
         "_orig_model_type": "unlimited-ocr",
         "sliding_window": 128,
         "sliding_window_size": 128,
         "moe_intermediate_size": 896,
         "n_group": 1,
         "n_routed_experts": 64,
         "n_shared_experts": 2,
         "num_attention_heads": 10,
         "num_experts_per_tok": 6,
         "num_hidden_layers": 12,
         "num_key_value_heads": 10,
         "projector_config": {
          "input_dim": 2048,
          "model_type": "mlp_projector",
          "n_embed": 1280,
          "projector_type": "linear"
         },
         "vision_config": {
          "image_size": 1024,
          "mlp_ratio": 3.7362,
          "model_name": "deeplip_b_l",
          "model_type": "vision",
          "width": {
           "clip-l-14-224": {
            "heads": 16,
            "image_size": 224,
            "layers": 24,
            "patch_size": 14,
            "width": 1024
           },
           "sam_vit_b": {
            "downsample_channels": [512, 1024],
            "global_attn_indexes": [2, 5, 8, 11],
            "heads": 12,
            "layers": 12,
            "width": 768
           }
          }
         },
         "vocab_size": 129280
        }
        """#

    fileprivate static let unlimitedOCRProcessorConfig = #"""
        {
         "add_special_token": false,
         "candidate_resolutions": [[1024, 1024]],
         "downsample_ratio": 4,
         "ignore_id": -100,
         "image_mean": [0.5, 0.5, 0.5],
         "image_std": [0.5, 0.5, 0.5],
         "image_token": "<image>",
         "mask_prompt": false,
         "normalize": true,
         "pad_token": "<｜▁pad▁｜>",
         "patch_size": 16,
         "processor_class": "DeepseekVLV2Processor",
         "sft_format": "deepseek"
        }
        """#
}
