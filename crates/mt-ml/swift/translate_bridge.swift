import Foundation
import Translation

// Translate bridge: reads JSON lines from stdin, writes translated text to stdout.
// Input:  {"texts": [...], "source": "en", "target": "pl", "batch_size": 8}
// Output: {"texts": [...], "error": null}

guard let input = readLine()?.data(using: .utf8),
      let request = try? JSONSerialization.jsonObject(with: input) as? [String: Any],
      let texts = request["texts"] as? [String] else {
    fputs("{\"error\": \"invalid input\"}", stderr)
    exit(1)
}

let source = request["source"] as? String ?? "en"
let target = request["target"] as? String ?? "pl"
let batchSize = request["batch_size"] as? Int ?? 8

let semaphore = DispatchSemaphore(value: 0)
var results: [String] = []
var error: String?

// macOS 26+ Translation framework
if #available(macOS 26, *) {
    Task {
        do {
            let translator = try await Translator(source: Locale(identifier: source),
                                                   target: Locale(identifier: target))
            for i in stride(from: 0, to: texts.count, by: batchSize) {
                let batch = Array(texts[i..<min(i + batchSize, texts.count)])
                for text in batch {
                    let result = try await translator.translate(text)
                    results.append(result)
                }
            }
        } catch let e {
            error = e.localizedDescription
        }
        semaphore.signal()
    }
    semaphore.wait()
} else {
    error = "macOS 26+ required for Translation framework"
    semaphore.signal()
}

if let err = error {
    fputs("{\"error\": \"\(err)\"}", stderr)
    exit(1)
}

let output = try? JSONSerialization.data(withJSONObject: ["texts": results])
if let data = output {
    print(String(data: data, encoding: .utf8) ?? "")
}
