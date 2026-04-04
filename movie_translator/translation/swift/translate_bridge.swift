import Foundation
import Translation

struct BatchRequest: Codable {
    let texts: [String]
    let source: String
    let target: String
}

struct BatchResponse: Codable {
    let translations: [String]?
    let elapsed_ms: Int?
    let error: String?
    let code: String?
}

@main
struct TranslateBridge {
    static func main() async {
        do {
            let inputData = FileHandle.standardInput.readDataToEndOfFile()
            let request = try JSONDecoder().decode(BatchRequest.self, from: inputData)

            let session = TranslationSession(
                installedSource: Locale.Language(identifier: request.source),
                target: Locale.Language(identifier: request.target)
            )

            let start = CFAbsoluteTimeGetCurrent()
            let batchRequests = request.texts.enumerated().map { i, text in
                TranslationSession.Request(sourceText: text, clientIdentifier: "\(i)")
            }
            let responses = try await session.translations(from: batchRequests)
            let elapsedMs = Int((CFAbsoluteTimeGetCurrent() - start) * 1000)

            let response = BatchResponse(
                translations: responses.map(\.targetText),
                elapsed_ms: elapsedMs,
                error: nil,
                code: nil
            )
            let output = try JSONEncoder().encode(response)
            FileHandle.standardOutput.write(output)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)

        } catch let error as TranslationError {
            let code: String
            let message: String
            switch error {
            case .notInstalled:
                code = "not_installed"
                message = "Translation languages not installed. Download via: System Settings > General > Language & Region > Translation Languages"
            case .unsupportedLanguagePairing:
                code = "unsupported"
                message = "Unsupported language pairing"
            default:
                code = "internal"
                message = "Translation error: \(error)"
            }
            let errResponse = BatchResponse(translations: nil, elapsed_ms: nil, error: message, code: code)
            if let out = try? JSONEncoder().encode(errResponse) {
                FileHandle.standardOutput.write(out)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
            Foundation.exit(1)

        } catch {
            let errResponse = BatchResponse(translations: nil, elapsed_ms: nil, error: "\(error)", code: "internal")
            if let out = try? JSONEncoder().encode(errResponse) {
                FileHandle.standardOutput.write(out)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
            Foundation.exit(1)
        }
    }
}
