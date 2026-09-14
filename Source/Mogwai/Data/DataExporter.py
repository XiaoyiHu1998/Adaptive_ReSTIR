from falcor import *
import os
import os.path as path
import json

baseDirectory = ""
subDirectory = ""


def writeJSON(dict: dict, path: str):
    jsonString = json.dumps(dict)
    with open(path, "w+") as file:
        file.write(jsonString)


def captureProfilerData(frameCount: int, m):
    assert(baseDirectory is not "")
    assert(subDirectory is not "")

    m.profiler.enabled = True
    m.profiler.startCapture()
    for frame in range(frameCount):
        m.renderFrame()
    capture = m.profiler.endCapture()
    m.profiler.enabled = False

    pathPrefix = f"{baseDirectory}/ProfilerCapture/{subDirectory}"
    if not path.exists(pathPrefix):
        os.mkdir(pathPrefix)
        
    filepath = f"{pathPrefix}/profilerCapture.json"
    if os.path.isfile(filepath):
        os.remove(filepath)

    writeJSON(capture, filepath)


# capture list of frames
def captureFrames(exitFrame: int, frameList: list, m):
    assert(baseDirectory is not "")
    assert(subDirectory is not "")

    directory = f"{baseDirectory}/FrameCapture/{subDirectory}"
    if not path.exists(directory):
        os.mkdir(directory)

    files = os.listdir(directory)
    if len(files) > 0:
        for file in files:
            os.remove(f"{directory}/{file}")

    m.clock.exitFrame = exitFrame
    m.frameCapture.outputDir = directory
    m.frameCapture.baseFilename = "Mogwai"
    m.frameCapture.addFrames(m.activeGraph, frameList)


# capture frames while clock is paused
def captureFramesPaused(frameCount: int, m, targetFrames: list = []):
    assert(baseDirectory is not "")
    assert(subDirectory is not "")

    directory = f"{baseDirectory}/FrameCapture/{subDirectory}"
    if not path.exists(directory):
        os.mkdir(directory)

    files = os.listdir(directory)
    if len(files) > 0:
        for file in files:
            os.remove(f"{directory}/{file}")

    m.frameCapture.outputDir = directory
    m.clock.pause()

    if len(targetFrames) == 0:
        for i in range(frameCount):
            renderFrame()
            m.frameCapture.baseFilename = f"Mogwai-{i:04d}"
            m.frameCapture.capture()
    else:
        for i in range(frameCount):
            renderFrame()
            if i in targetFrames:
                m.frameCapture.baseFilename = f"Mogwai-{i:04d}"
                m.frameCapture.capture()

    m.clock.play()


# Timing Capture
def captureTiming(m):
    assert(baseDirectory is not "")
    assert(subDirectory is not "")

    directory = f"{baseDirectory}/TimeCapture/{subDirectory}"
    if not path.exists(directory):
        os.mkdir(directory)

    filepath = f"{directory}/timingCapture.csv"
    if os.path.isfile(filepath):
        os.remove(filepath)

    m.timingCapture.captureFrameTime(filepath)