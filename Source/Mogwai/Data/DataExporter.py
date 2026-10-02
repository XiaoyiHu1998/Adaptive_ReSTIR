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
def captureFrames(exitFrame: int, frameList: list, m):
    assert(baseDirectory is not "")
    assert(sceneName is not "")
    assert(runName is not "")

    sceneDirectory = f"{baseDirectory}/Captures/{sceneName}"
    runDirectory = f"{baseDirectory}/Captures/{sceneName}/{runName}"

    if not path.exists(sceneDirectory):
        os.mkdir(sceneDirectory)

    if not path.exists(runDirectory):
        os.mkdir(runDirectory)
        
    files = os.listdir(runDirectory)
    if len(files) > 0:
        for file in files:
            os.remove(f"{runDirectory}/{file}")

    m.clock.exitFrame = exitFrame
    m.frameCapture.outputDir = runDirectory
    m.frameCapture.baseFilename = "Mogwai"
    m.frameCapture.addFrames(m.activeGraph, frameList)


# capture frames while clock is paused
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

    files = os.listdir(runDirectory)
    if len(files) > 0:
        for file in files:
            os.remove(f"{runDirectory}/{file}")

    m.frameCapture.outputDir = runDirectory
    m.clock.pause()

    if len(targetFrames) == 0:
        for i in range(frameCount):
            m.renderFrame()
            m.frameCapture.baseFilename = f"Mogwai-{i:04d}"
            m.frameCapture.capture()
    else:
        for i in range(frameCount):
            m.renderFrame()
            if i in targetFrames:
                m.frameCapture.baseFilename = f"Mogwai-{i:04d}"
                m.frameCapture.capture()

    m.clock.play()


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