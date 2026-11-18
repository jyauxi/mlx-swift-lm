// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@_spi(Testing) @testable import MLXVLM

final class DeepseekOCRVisionTests: XCTestCase {

    func testSAMEncoderMatchesExpectedFeatureShape() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.samFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 16, 16, 1024])
    }

    func testClipFusionProducesExpectedPreProjectorShape() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.fusedVisionFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 256, 2048])
    }

    func testProjectorOutputsLanguageHiddenSize() throws {
        let model = try makeModel()
        let pixels = zeros([1, 1024, 1024, 3], type: Float.self)

        let features = model.projectedImageFeaturesForTesting(pixels)

        XCTAssertEqual(features.shape, [1, 256, 1280])
    }

    func testSanitizeRemapsSAMWeightsAndPreservesShapes() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "model.sam_model.patch_embed.proj.weight": zeros([768, 3, 16, 16], type: Float.self),
            "model.sam_model.blocks.0.attn.qkv.weight": zeros([2304, 768], type: Float.self),
            "model.sam_model.neck.0.weight": zeros([256, 768, 1, 1], type: Float.self),
            "model.sam_model.neck.2.weight": zeros([256, 256, 3, 3], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["sam_model.patch_embed.proj.weight"]?.shape, [768, 16, 16, 3])
        XCTAssertEqual(sanitized["sam_model.layers.0.attn.qkv.weight"]?.shape, [2304, 768])
        XCTAssertEqual(sanitized["sam_model.neck.conv1.weight"]?.shape, [256, 1, 1, 768])
        XCTAssertEqual(sanitized["sam_model.neck.conv2.weight"]?.shape, [256, 3, 3, 256])
    }

    func testSanitizeRemapsClipAndProjectorWeights() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "model.vision_model.embeddings.class_embedding": zeros([1024], type: Float.self),
            "model.vision_model.pre_layrnorm.weight": zeros([1024], type: Float.self),
            "model.projector.layers.weight": zeros([1280, 2048], type: Float.self),
            "model.projector.layers.bias": zeros([1280], type: Float.self),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["vision_model.embeddings.class_embedding"]?.shape, [1024])
        XCTAssertEqual(sanitized["vision_model.pre_layrnorm.weight"]?.shape, [1024])
        XCTAssertEqual(sanitized["projector.layers.weight"]?.shape, [1280, 2048])
        XCTAssertEqual(sanitized["projector.layers.bias"]?.shape, [1280])
    }

    func testClipFeedForwardUsesQuickGelu() throws {
        // Golden values through float32 matmuls, which TF32 (neural accelerators) moves by
        // ~1e-3 relative. The CPU pins IEEE float32; see MatmulPrecision.
        try Device.withDefaultDevice(.cpu) { try clipFeedForwardUsesQuickGelu() }
    }

    private func clipFeedForwardUsesQuickGelu() throws {
        let model = try makeModel()
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let prefix = "vision_model.transformer.layers.0.mlp."
        let fc1Weight = try XCTUnwrap(weights[prefix + "fc1.weight"])
        let fc1Bias = try XCTUnwrap(weights[prefix + "fc1.bias"])
        let fc2Weight = try XCTUnwrap(weights[prefix + "fc2.weight"])
        let fc2Bias = try XCTUnwrap(weights[prefix + "fc2.bias"])
        let x = sin(MLXArray((0 ..< 2048).map { Float($0) / 37 })).reshaped(1, 2, 1024)

        let hidden = matmul(x, fc1Weight.T) + fc1Bias
        let expected = matmul(hidden * sigmoid(1.702 * hidden), fc2Weight.T) + fc2Bias

        XCTAssertTrue(allClose(model.clipFeedForwardForTesting(x), expected, atol: 1e-4).item())
    }

    /// Expected values are mlx-vlm 0.8.0 `bicubic_interpolate(..., antialias=True)`, the
    /// authors' `F.interpolate(mode="bicubic", antialias=True)`, on a 16 × 16 ramp.
    func testClipPositionEmbeddingResizeIsAntialiasedBicubic() throws {
        // Golden values through float32 matmuls, which TF32 (neural accelerators) moves by
        // ~1e-3 relative. The CPU pins IEEE float32; see MatmulPrecision.
        try Device.withDefaultDevice(.cpu) { try clipPositionEmbeddingResizeIsAntialiasedBicubic() }
    }

    private func clipPositionEmbeddingResizeIsAntialiasedBicubic() throws {
        let model = try makeModel()
        let ramp = MLXArray((0 ..< 256).map { Float($0) })
        let grid = broadcast(ramp.reshaped(1, 256, 1), to: [1, 256, 1024])
        let cls = full([1, 1, 1024], values: Float(-1))
        model.update(
            parameters: ModuleParameters.unflattened([
                ("vision_model.embeddings.position_embedding", concatenated([cls, grid], axis: 1))
            ]))

        let resized = model.clipPositionEmbeddingForTesting(gridSize: 10)

        XCTAssertEqual(resized.shape, [1, 101, 1024])
        XCTAssertEqual(resized[0, 0, 0].item(Float.self), -1, "the class token is not resized")
        let output = resized[0, 1..., 0].reshaped(10, 10)
        let row0: [Float] = [
            5.540069, 7.092096, 8.714179, 10.310905, 11.915564, 13.512805, 15.117456,
            16.714176, 18.336273, 19.888298,
        ]
        let row5: [Float] = [
            133.10384, 134.65585, 136.27794, 137.87462, 139.47935, 141.07657, 142.68124,
            144.27794, 145.90002, 147.45206,
        ]
        XCTAssertTrue(allClose(output[0], MLXArray(row0), atol: 1e-3).item(Bool.self))
        XCTAssertTrue(allClose(output[5], MLXArray(row5), atol: 1e-3).item(Bool.self))
        let corners = [output[0, 0], output[0, 9], output[9, 0], output[9, 9]]
        XCTAssertTrue(
            allClose(
                stacked(corners), MLXArray([5.540069, 19.888298, 235.11171, 249.45993] as [Float]),
                atol: 1e-3
            ).item(Bool.self))
    }

    /// Expected values are mlx-vlm 0.8.0 `bicubic_interpolate(..., antialias=True)` on a
    /// 64 × 64 grid of `sin(k / 7)`, resized for a 640 gundam tile.
    func testSAMPositionEmbeddingResizeIsAntialiasedBicubic() throws {
        // Golden values through float32 matmuls, which TF32 (neural accelerators) moves by
        // ~1e-3 relative. The CPU pins IEEE float32; see MatmulPrecision.
        try Device.withDefaultDevice(.cpu) { try sAMPositionEmbeddingResizeIsAntialiasedBicubic() }
    }

    private func sAMPositionEmbeddingResizeIsAntialiasedBicubic() throws {
        let model = try makeModel()
        let wave = sin(MLXArray((0 ..< 4096).map { Float($0) / 7 }))
        let grid = broadcast(wave.reshaped(1, 64, 64, 1), to: [1, 64, 64, 768])
        model.update(parameters: ModuleParameters.unflattened([("sam_model.pos_embed", grid)]))

        let resized = try XCTUnwrap(model.samPositionEmbeddingForTesting(gridSize: 40))
        let same = try XCTUnwrap(model.samPositionEmbeddingForTesting(gridSize: 64))

        XCTAssertEqual(resized.shape, [1, 40, 40, 768])
        let expected: [Float] = [0.12154, 0.160875, 0.194009, 0.215981, 0.227155]
        XCTAssertTrue(
            allClose(resized[0, 0, ..<5, 0], MLXArray(expected), atol: 1e-4).item(Bool.self))
        XCTAssertEqual(resized[0, 17, 23, 0].item(Float.self), 0.060723055, accuracy: 1e-4)
        XCTAssertTrue(
            allClose(same[0, 0..., 0..., 0], wave.reshaped(64, 64)).item(Bool.self),
            "the same size is the identity")
    }

    /// SAM resizes relative positions with half-pixel linear interpolation; expected values
    /// are mlx-vlm 0.8.0 `get_rel_pos` and the authors' `F.interpolate(mode="linear")`.
    func testRelativePositionResizeIsHalfPixelLinear() {
        let squares = MLXArray((0 ..< 9).map { Float($0 * $0) }).reshaped(9, 1)
        let down: [Float] = [0.4, 5.0, 16.0, 33.8, 58.0]
        let up: [Float] = [
            0, 0.538462, 1.692308, 3.769231, 7.076923, 11.153848, 16, 22.230772, 29.23077,
            37.000004, 46, 55.92308, 64,
        ]

        for expected in [down, up] {
            let resized = resizedRelativePositions(squares, length: expected.count)
            XCTAssertEqual(resized.shape, [expected.count, 1])
            XCTAssertTrue(
                allClose(resized.squeezed(axis: 1), MLXArray(expected), atol: 1e-4).item(Bool.self),
                "\(resized.squeezed(axis: 1).asArray(Float.self))")
        }
    }

    /// The resized relative positions keep the weights' dtype, as mlx-vlm casts them back: a
    /// float32 bias on a bfloat16 query is a mask SDPA rejects.
    func testRelativePositionResizeKeepsTheWeightDtype() {
        let relPos = MLXArray((0 ..< 27).map { Float($0) }).reshaped(27, 1).asType(.bfloat16)
        XCTAssertEqual(resizedRelativePositions(relPos, length: 79).dtype, .bfloat16)
    }

    private func makeModel() throws -> DeepseekOCR {
        let config = try JSONDecoder().decode(
            DeepseekOCRConfiguration.self,
            from: Data(Self.configJSON.utf8))
        return DeepseekOCR(config)
    }

    private static let configJSON = #"""
        {
         "model_type": "deepseekocr",
         "vision_config": {
          "hidden_size": 768,
          "output_channels": 256,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "image_size": 1024,
          "patch_size": 16,
          "global_attn_indexes": [2, 5, 8, 11],
          "mlp_dim": 3072
         },
         "language_config": {
          "vocab_size": 129280,
          "hidden_size": 1280,
          "intermediate_size": 6848,
          "num_hidden_layers": 12,
          "num_attention_heads": 10,
          "num_key_value_heads": 10,
          "max_position_embeddings": 8192
         }
        }
        """#
}
