import Foundation
import Speech

// Transcribe bridge: takes audio file path + locale, outputs JSON segments to stdout.
// Usage: transcribe_bridge <audio.wav> <locale>
// Output: {"segments": [{"text": "...", "start_ms": N, "end_ms": N}, ...]}

guard CommandLine.arguments.count > 2 else {
    fputs("Usage: transcribe_bridge <audio.wav> <locale>", stderr)
    exit(1)
}

let audioPath = CommandLine.arguments[1]
let locale = CommandLine.arguments[2]
let audioURL = URL(fileURLWithPath: audioPath)

let semaphore = DispatchSemaphore(value: 0)
var segments: [[String: Any]] = []
var error: String?

if #available(macOS 26, *) {
    let recognizer = SpeechAnalyzer()

    Task {
        do {
            let results = try await recognizer.transcribe(audioURL, locale: Locale(identifier: locale))
            for segment in results {
                segments.append([
                    "text": segment.text,
                    "start_ms": Int(segment.startTime.seconds * 1000),
                    "end_ms": Int(segment.endTime.seconds * 1000),
                ])
            }
        } catch let e {
            error = e.localizedDescription
        }
        semaphore.signal()
    }
} else {
    error = "macOS 26+ required for SpeechAnalyzer"
    semaphore.signal()
}

semaphore.wait()

if let err = error {
    fputs("{\"error\": \"\(err)\"}", stderr)
    exit(1)
}

let output = try? JSONSerialization.data(withJSONObject: ["segments": segments])
if let data = output {
    print(String(data: data, encoding: .utf8) ?? "")
}
