// macOS 시스템(또는 특정 앱) 소리를 ScreenCaptureKit 으로 받아
// 모노 float32 PCM 을 stdout 으로 내보내는 도우미. jp_subtitle 이 subprocess 로 실행한다.
//
//   sck_audio              모든 소리
//   sck_audio --app maum   이름(또는 번들 ID)에 maum 이 들어간 앱 소리만
//   sck_audio --list       소리를 낼 수 있는 실행 중인 앱 목록
//
// stderr 로 상태를 한 줄씩 알린다:
//   FORMAT <샘플레이트>   첫 오디오가 오기 직전 한 번
//   READY                캡처 시작
//   PERMISSION           화면 및 시스템 오디오 녹음 권한이 없음
//   그 밖의 줄은 사람이 읽는 오류 메시지
//
// 빌드: swiftc -O -target arm64-apple-macos13 mac/sck_audio.swift -o build/sck_audio

import CoreMedia
import Foundation
import ScreenCaptureKit

func say(_ line: String) {
    FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
}

func fail(_ message: String, code: Int32) -> Never {
    say(message)
    exit(code)
}

final class AudioOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private var announced = false
    private var mono: [Float] = []

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferIsValid(sampleBuffer),
              let block = CMSampleBufferGetDataBuffer(sampleBuffer),
              let format = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee
        else { return }

        guard asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0, asbd.mBitsPerChannel == 32 else {
            fail("지원하지 않는 오디오 형식입니다 (flags=\(asbd.mFormatFlags), bits=\(asbd.mBitsPerChannel))", code: 7)
        }
        if !announced {
            say("FORMAT \(Int(asbd.mSampleRate))")
            announced = true
        }

        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let frames = CMSampleBufferGetNumSamples(sampleBuffer)
        let length = CMBlockBufferGetDataLength(block)
        guard frames > 0, length >= frames * channels * 4 else { return }

        var raw = [Float](repeating: 0, count: length / 4)
        let status = raw.withUnsafeMutableBytes { buf in
            CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: buf.baseAddress!)
        }
        guard status == kCMBlockBufferNoErr else { return }

        // 여러 채널이면 평균을 내서 모노로 만든다
        if channels == 1 {
            mono = Array(raw.prefix(frames))
        } else {
            let planar = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
            mono = [Float](repeating: 0, count: frames)
            for i in 0..<frames {
                var sum: Float = 0
                for c in 0..<channels {
                    sum += planar ? raw[c * frames + i] : raw[i * channels + c]
                }
                mono[i] = sum / Float(channels)
            }
        }
        mono.withUnsafeBytes { buf in
            _ = fwrite(buf.baseAddress, 1, buf.count, stdout)
        }
        fflush(stdout)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail("캡처가 중단되었습니다: \(error.localizedDescription)", code: 4)
    }
}

var appName: String? = nil
var listOnly = false
var args = CommandLine.arguments.dropFirst().makeIterator()
while let arg = args.next() {
    switch arg {
    case "--list": listOnly = true
    case "--app": appName = args.next()
    default: fail("알 수 없는 옵션: \(arg)", code: 64)
    }
}

let output = AudioOutput()
do {
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    if listOnly {
        let names = Set(content.applications.map { "\($0.applicationName)\t\($0.bundleIdentifier)" })
        for name in names.sorted() { print(name) }
        exit(0)
    }
    guard let display = content.displays.first else { fail("디스플레이를 찾지 못했습니다", code: 5) }

    let filter: SCContentFilter
    if let name = appName, !name.isEmpty {
        let apps = content.applications.filter {
            $0.applicationName.localizedCaseInsensitiveContains(name)
                || $0.bundleIdentifier.localizedCaseInsensitiveContains(name)
        }
        if apps.isEmpty { fail("'\(name)' 앱을 찾지 못했습니다. 그 앱을 먼저 실행하세요.", code: 6) }
        filter = SCContentFilter(display: display, including: apps, exceptingWindows: [])
    } else {
        filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
    }

    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.sampleRate = 48000
    config.channelCount = 1
    // 화면은 필요 없으므로 최소 크기, 최소 빈도로
    config.width = 2
    config.height = 2
    config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

    let stream = SCStream(filter: filter, configuration: config, delegate: output)
    try stream.addStreamOutput(output, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio"))
    try await stream.startCapture()
    say("READY")
    while true {
        try await Task.sleep(nanoseconds: 3_600_000_000_000)
    }
} catch {
    let nsError = error as NSError
    if nsError.domain == SCStreamErrorDomain && nsError.code == SCStreamError.Code.userDeclined.rawValue {
        fail("PERMISSION", code: 3)
    }
    fail("캡처를 시작하지 못했습니다: \(error.localizedDescription)", code: 2)
}
