// Copyright © 2026 Apple Inc.

import MLXLMCommon
import XCTest

final class ProcessorTypeRegistryTests: XCTestCase {

    func testContainsReflectsRegisteredProcessorTypes() async {
        let registry = ProcessorTypeRegistry(creators: [
            "TestProcessor": { _, _ in TestProcessor() }
        ])

        let containsRegistered = await registry.contains("TestProcessor")
        let containsUnregistered = await registry.contains("UnregisteredProcessor")

        XCTAssertTrue(containsRegistered)
        XCTAssertFalse(containsUnregistered)
    }

    func testContainsAgreesWithCreateModelWithoutInstantiating() async {
        let registry = ProcessorTypeRegistry()
        await registry.registerProcessorType("RegisteredProcessor") { _, _ in TestProcessor() }

        let contains = await registry.contains("RegisteredProcessor")
        XCTAssertTrue(contains)

        do {
            _ = try await registry.createModel(
                configuration: Data(), processorType: "UnregisteredProcessor",
                tokenizer: TestTokenizer())
            XCTFail("Expected createModel to throw for an unregistered processor type")
        } catch let error as ModelFactoryError {
            guard case .unsupportedProcessorType("UnregisteredProcessor") = error else {
                return XCTFail("Expected unsupportedProcessorType, got \(error)")
            }
        } catch {
            XCTFail("Expected a ModelFactoryError, got \(error)")
        }
    }
}

private struct TestProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("not exercised by ProcessorTypeRegistryTests")
    }
}
