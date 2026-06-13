import Foundation
import Vision
import Quartz

guard CommandLine.arguments.count > 1 else {
    fputs("Usage: ocr_bridge <image-path>", stderr)
    exit(1)
}
let imagePath = CommandLine.arguments[1]
let url = URL(fileURLWithPath: imagePath)
guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
    exit(0)
}

let request = VNRecognizeTextRequest { request, error in
    if let error = error {
        fputs("Vision error: \(error.localizedDescription)", stderr)
        exit(1)
    }
    guard let observations = request.results as? [VNRecognizedTextObservation] else { exit(0) }
    var lines: [String] = []
    for obs in observations {
        if let candidate = obs.topCandidates(1).first {
            lines.append(candidate.string)
        }
    }
    print(lines.joined(separator: "\n"))
}
request.recognitionLevel = .accurate
request.recognitionLanguages = ["en"]
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try? handler.perform([request])
