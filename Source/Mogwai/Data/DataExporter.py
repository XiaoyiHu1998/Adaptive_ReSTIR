from falcor import *
import os
import os.path as path
import json

baseDirectory = ""
sceneName = ""
runName = ""


def writeJSON(dict: dict, path: str):
    jsonString = json.dumps(dict)
    with open(path, "w+") as file:
        file.write(jsonString)


def captureProfilerData(frameCount: int, m):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    m.profiler.enabled = True
    m.profiler.paused = False
    m.profiler.startCapture()
    for frame in range(frameCount):
        m.renderFrame()
    capture = m.profiler.endCapture()
    m.profiler.enabled = False
    m.profiler.paused = True

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)
        
    filepath = f"{runDirectory}/profilerCapture.json"
    if os.path.isfile(filepath):
        os.remove(filepath)

    writeJSON(capture, filepath)


# capture list of frames
def captureAnimationFrames(startFrame: int, frameCount: int, m):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)
        
    files = [file for file in os.listdir(runDirectory) if path.isfile(path.join(runDirectory, file))]
    if len(files) > 0:
        for file in files:
            os.remove(f"{runDirectory}/{file}")

    m.frameCapture.outputDir = runDirectory
    m.clock.pause()

    if startFrame > 0:
        m.clock.step(frames=startFrame)

    for i in range(startFrame, startFrame + frameCount):
        m.renderFrame()
        m.frameCapture.baseFilename = f"{sceneName}_{runName}_{i:04d}"
        m.frameCapture.capture()
        m.clock.step(frames=1)


# capture frames with paused clock
def captureFramesPaused(frameCount: int, m, targetFrames: list = []):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)

    files = [file for file in os.listdir(runDirectory) if path.isfile(path.join(runDirectory, file))]
    if len(files) > 0:
        for file in files:
            os.remove(f"{runDirectory}/{file}")

    m.frameCapture.outputDir = runDirectory
    m.clock.pause()

    if len(targetFrames) == 0:
        for i in range(frameCount):
            m.renderFrame()
            m.frameCapture.baseFilename = f"{sceneName}_{runName}_{i:04d}"
            m.frameCapture.capture()
    else:
        for i in range(frameCount):
            m.renderFrame()
            if i in targetFrames:
                m.frameCapture.baseFilename = f"{sceneName}_{runName}_{i:04d}"
                m.frameCapture.capture()

    m.clock.play()


# capture frames while clock is paused
def captureReferenceFrames(frameCount: int, accumulationCount: int, m, startFrame = 0):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)

    existingFiles = os.listdir(runDirectory)
    existingFrameIndices = sorted([int(existingFile.split(".")[0].split("_")[-1]) for existingFile in existingFiles if existingFile.split(".")[-1] == "png"])

    if len(existingFrameIndices) == frameCount and existingFrameIndices[0] == startFrame and existingFrameIndices[-1] == startFrame + frameCount - 1:
        return

    m.frameCapture.outputDir = runDirectory
    m.clock.pause()

    if startFrame > 0:
        m.clock.step(frames=startFrame)

    for frameIndex in range(startFrame, startFrame + frameCount):
        if frameIndex in existingFrameIndices:
            m.clock.step(frames=1)
            continue

        for _ in range(accumulationCount):
            m.renderFrame()

        m.frameCapture.baseFilename = f"Reference_{frameIndex:04d}"
        m.frameCapture.capture()
        m.clock.step(frames=1)


# Timing Capture
def captureTiming(m):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)

    filepath = f"{runDirectory}/timingCapture.csv"
    if os.path.isfile(filepath):
        os.remove(filepath)

    m.timingCapture.captureFrameTime(filepath)