// Offline render through Apple's AUSoundIsolation (the voice-isolation model
// behind Final Cut Pro / Logic's "Voice Isolation"). macOS 13+.
//
//   soundisolation <input.wav> <output.wav>
//
// Writes 32-bit float WAV, same rate/channels as the input, latency-compensated.

import AVFoundation
import AudioToolbox
import Foundation

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func fourCC(_ s: String) -> OSType {
    s.utf8.reduce(0) { ($0 << 8) | OSType($1) }
}

let args = CommandLine.arguments
guard args.count == 3 else { fail("usage: soundisolation <input.wav> <output.wav>") }

do {
    let inFile = try AVAudioFile(forReading: URL(fileURLWithPath: args[1]))
    let format = inFile.processingFormat

    let desc = AudioComponentDescription(
        componentType: kAudioUnitType_Effect,
        componentSubType: fourCC("vois"),
        componentManufacturer: kAudioUnitManufacturer_Apple,
        componentFlags: 0, componentFlagsMask: 0)
    guard AudioComponentFindNext(nil, [desc]) != nil else {
        fail("AUSoundIsolation isn't available on this Mac (needs macOS 13 or later).")
    }
    let effect = AVAudioUnitEffect(audioComponentDescription: desc)

    // Make sure the unit is 100% wet; mixing is done by the Python side.
    if let tree = effect.auAudioUnit.parameterTree {
        for p in tree.allParameters {
            let name = (p.identifier + " " + p.displayName).lowercased()
            if name.contains("mix") { p.value = p.maxValue }
        }
    }

    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    engine.attach(player)
    engine.attach(effect)
    engine.connect(player, to: effect, format: format)
    engine.connect(effect, to: engine.mainMixerNode, format: format)

    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
    try engine.start()
    player.scheduleFile(inFile, at: nil)
    player.play()

    let latency = AVAudioFramePosition((effect.auAudioUnit.latency * format.sampleRate).rounded())
    let total = inFile.length + latency
    var toSkip = latency

    let outFile = try AVAudioFile(
        forWriting: URL(fileURLWithPath: args[2]),
        settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ],
        commonFormat: .pcmFormatFloat32, interleaved: false)

    guard let buf = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat,
                                     frameCapacity: engine.manualRenderingMaximumFrameCount)
    else { fail("could not allocate render buffer") }

    while engine.manualRenderingSampleTime < total {
        let remaining = total - engine.manualRenderingSampleTime
        let frames = AVAudioFrameCount(min(Int64(buf.frameCapacity), remaining))
        switch try engine.renderOffline(frames, to: buf) {
        case .success:
            var n = Int(buf.frameLength)
            if toSkip > 0 {
                let skip = Int(min(Int64(n), toSkip))
                toSkip -= Int64(skip)
                n -= skip
                if n == 0 { continue }
                for ch in 0..<Int(buf.format.channelCount) {
                    let d = buf.floatChannelData![ch]
                    d.update(from: d + skip, count: n)
                }
                buf.frameLength = AVAudioFrameCount(n)
            }
            try outFile.write(from: buf)
        case .insufficientDataFromInputNode, .cannotDoInCurrentContext:
            continue
        case .error:
            fail("render error")
        @unknown default:
            fail("unknown render status")
        }
    }
    engine.stop()
} catch {
    fail("error: \(error)")
}
