// Apple SpeechAnalyzer/SpeechTranscriber file transcription (macOS 26+).
// Usage: apple_speech <wav> <locale e.g. en-US|ja-JP>
// Emits {"segments":[{"start_ms","end_ms","text"}]} on stdout.

import AVFoundation
import Foundation
import Speech

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

guard CommandLine.arguments.count >= 3 else { fail("usage: apple_speech <wav> <locale>") }
let wavPath = CommandLine.arguments[1]
let localeID = CommandLine.arguments[2]

@available(macOS 26, *)
func run() async throws {
    let locale = Locale(identifier: localeID)
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange]
    )

    // Ensure the on-device model asset for this locale is installed.
    let supported = await SpeechTranscriber.supportedLocales
    if !supported.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
        fail("locale \(localeID) not supported by SpeechTranscriber")
    }
    if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        try await req.downloadAndInstall()
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])

    let url = URL(fileURLWithPath: wavPath)
    let audioFile = try AVAudioFile(forReading: url)

    // Collect results concurrently while we feed the file.
    var segs: [[String: Any]] = []
    let resultsTask = Task {
        for try await result in transcriber.results {
            let text = String(result.text.characters)
            var startMs = -1, endMs = 0
            for run in result.text.runs {
                if let range = run.audioTimeRange {
                    let s = Int(range.start.seconds * 1000)
                    let e = Int((range.start + range.duration).seconds * 1000)
                    if startMs < 0 { startMs = s }
                    endMs = max(endMs, e)
                }
            }
            if startMs < 0 { startMs = 0 }
            segs.append(["start_ms": startMs, "end_ms": endMs, "text": text])
        }
    }

    _ = try await analyzer.analyzeSequence(from: audioFile)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    try await resultsTask.value

    let out: [String: Any] = ["segments": segs]
    let data = try JSONSerialization.data(withJSONObject: out, options: [])
    FileHandle.standardOutput.write(data)
}

if #available(macOS 26, *) {
    let sem = DispatchSemaphore(value: 0)
    Task {
        do { try await run() } catch { fail("error: \(error)") }
        sem.signal()
    }
    sem.wait()
} else {
    fail("requires macOS 26+")
}
