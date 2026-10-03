import os
import os.path as path
import subprocess
import json
import sys
import matplotlib.pyplot as plt
import statistics as st
from concurrent.futures import ThreadPoolExecutor


def executeImageCompareCall(commandList):
    return subprocess.run(commandList, capture_output=True, text=True).stdout.strip()


def ExportErrorMetrics():
    baseFolder = sys.argv[1]
    ImageComparePath = sys.argv[2]

    capturesPath = path.join(baseFolder, "Captures")
    scenes = [scene for scene in os.listdir(capturesPath) if path.isdir(path.join(capturesPath, scene))]

    runLogDict = dict()
    runLogDictPath = path.join(capturesPath, "runLogs.json")
    if path.exists(runLogDictPath):
        with open(runLogDictPath, "r") as file:
            runLogDict = json.load(file)

    for scene in scenes:
        scenePath = path.join(capturesPath, scene)
        runs = [run for run in os.listdir(scenePath) if path.isdir(path.join(scenePath, run)) and run != "Reference"]

        referencePath = path.join(scenePath, "Reference")
        referenceFrames = [path.join(referencePath, referenceImage) for referenceImage in os.listdir(referencePath) if referenceImage.split(".")[-1] == "png"]
        referenceFrames = sorted(referenceFrames)
        isAnimatedScene = len(referenceFrames) > 1

        if scene not in runLogDict.keys():
            runLogDict[scene] = dict()

        for run in runs:
            runPath = path.join(scenePath, run)

            if run not in runLogDict[scene].keys():
                runLogDict[scene][run] = path.getmtime(runPath)
            else:
                if runLogDict[scene][run] == path.getmtime(runPath):
                    print(f"skipped comparing images in {runPath}\t")
                    continue

            print(f"comparing images in {runPath}\t")
            runLogDict[scene][run] = path.getmtime(runPath)

            runFrames = [path.join(runPath, image) for image in os.listdir(runPath) if image.split(".")[-1] == "png"]
            runFrames = sorted(runFrames)
            
            errorsMAE = []
            errorsMSE = []
            errorsRMSE = []
            errorsMAPE = []
            
            maeCalls = []
            mseCalls = []
            rmseCalls = []
            mapeCalls = []

            # print(f"current frame: {runFrames[0]}", end="\r")
            for frameIndex in range(len(runFrames)):
                # print(f"current frame: {runFrames[frameIndex]}", end="\r")

                runFrame = runFrames[frameIndex]
                referenceFrame = referenceFrames[frameIndex] if isAnimatedScene else referenceFrames[0]

                # maeCall = [ImageComparePath, "-m", "mae", runFrame, referenceFrame]
                # mseCall = [ImageComparePath, "-m", "mse", runFrame, referenceFrame]
                # rmseCall = [ImageComparePath, "-m", "rmse", runFrame, referenceFrame]
                # mapeCall = [ImageComparePath, "-m", "mape", runFrame, referenceFrame]

                # errorsMAE.append(subprocess.run(maeCall, capture_output=True, text=True).stdout.strip())
                # errorsMSE.append(subprocess.run(mseCall, capture_output=True, text=True).stdout.strip())
                # errorsRMSE.append(subprocess.run(rmseCall, capture_output=True, text=True).stdout.strip())
                # errorsMAPE.append(subprocess.run(mapeCall, capture_output=True, text=True).stdout.strip())

                maeCall = [ImageComparePath, "-m", "mae", runFrame, referenceFrame]
                mseCall = [ImageComparePath, "-m", "mse", runFrame, referenceFrame]
                rmseCall = [ImageComparePath, "-m", "rmse", runFrame, referenceFrame]
                mapeCall = [ImageComparePath, "-m", "mape", runFrame, referenceFrame]

                maeCalls.append(maeCall)
                mseCalls.append(mseCall)
                rmseCalls.append(rmseCall)
                mapeCalls.append(mapeCall)

            with ThreadPoolExecutor(max_workers=16) as executor:
                errorsMAE = list(executor.map(executeImageCompareCall, maeCalls))
                errorsMSE = list(executor.map(executeImageCompareCall, mseCalls))
                errorsRMSE = list(executor.map(executeImageCompareCall, rmseCalls))
                errorsMAPE = list(executor.map(executeImageCompareCall, mapeCalls))

            exportDict = dict()
            exportDict["frameCount"] = len(runFrames)
            exportDict["mae"] = errorsMAE
            exportDict["mse"] = errorsMSE
            exportDict["rmse"] = errorsRMSE
            exportDict["mape"] = errorsMAPE

            with open(path.join(runPath, "errors.json"), "w+") as file:
                file.write(json.dumps(exportDict))

            print(f"exported frame error data to {path.join(runPath, "errors.json")}\t")

    with open(runLogDictPath, "w+") as file:
        file.write(json.dumps(runLogDict))

    print("Finished exporting frame error data\t\n")


def ExportAverageErrorMetrics():
    baseFolder = sys.argv[1]
    # ImageComparePath = sys.argv[2]

    capturesPath = path.join(baseFolder, "Captures")
    scenes = [scene for scene in os.listdir(capturesPath) if path.isdir(path.join(capturesPath, scene))]

    dataDict = dict()
    for scene in scenes:
        scenePath = path.join(capturesPath, scene)
        runs = [run for run in os.listdir(scenePath) if path.isdir(path.join(scenePath, run)) and run != "Reference"]

        for run in runs:
            print(f"Gathering image error data from {scene}\\{run}")
            runPath = path.join(scenePath, run)
            restirScheme = run.split("_")[0]

            errorDict = dict()
            with open(path.join(runPath, "errors.json"), "r") as file:
                errorDict = json.load(file)

            if scene not in dataDict.keys():
                dataDict[scene] = dict()

            if restirScheme not in dataDict[scene].keys():
                dataDict[scene][restirScheme] = [errorDict]
            else:
                dataDict[scene][restirScheme].append(errorDict)

    exportDict = dict()
    for scene in dataDict.keys():
        exportDict[scene] = dict()

        for restirScheme in dataDict[scene].keys():
            print(f"Generating average error data for {scene}\\{restirScheme}")
            dictList = dataDict[scene][restirScheme]
            dictCount = len(dictList)
            minFrameCount = sys.maxsize

            for errorDict in dictList:
                minFrameCount = min(minFrameCount, errorDict["frameCount"])
    
            averageMAE = []
            averageMSE = []
            averageRMSE = []
            averageMAPE = []

            for frameIndex in range(minFrameCount):
                mae = 0.0
                mse = 0.0
                rmse = 0.0
                mape = 0.0
    
                for dictIndex in range(dictCount):
                    mae += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mae"][frameIndex])
                    mse += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mse"][frameIndex])
                    rmse += (1.0 / float(dictCount)) * float(dictList[dictIndex]["rmse"][frameIndex])
                    mape += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mape"][frameIndex])
    
                averageMAE.append(mae)
                averageMSE.append(mse)
                averageRMSE.append(rmse)
                averageMAPE.append(mape)

            averageErrorDict = dict()
            averageErrorDict["frameCount"] = minFrameCount
            averageErrorDict["mae"] = averageMAE
            averageErrorDict["mse"] = averageMSE
            averageErrorDict["rmse"] = averageRMSE
            averageErrorDict["mape"] = averageMAPE

            exportDict[scene][restirScheme] = averageErrorDict

    print("Exporting average error data", end="\r")
    with open(path.join(capturesPath, "errors.json"), "w") as file:
        file.write(json.dumps(exportDict))
        
    print("Exported average error data\t\n")


def ExportAverageProfilerData():
    baseFolder = sys.argv[1]
    # ImageComparePath = sys.argv[2]

    capturesPath = path.join(baseFolder, "Captures")
    scenes = [scene for scene in os.listdir(capturesPath) if path.isdir(path.join(capturesPath, scene))]

    dataDict = dict()
    for scene in scenes:
        scenePath = path.join(capturesPath, scene)
        runs = [run for run in os.listdir(scenePath) if path.isdir(path.join(scenePath, run)) and run != "Reference"]

        for run in runs:
            print(f"Gathering profiler data from {scene}\\{run}")
            runPath = path.join(scenePath, run)
            restirScheme = run.split("_")[0]

            profilerDict = dict()
            with open(path.join(runPath, "profilerCapture.json"), "r") as file:
                profilerDict = json.load(file)

            if scene not in dataDict.keys():
                dataDict[scene] = dict()

            if restirScheme not in dataDict[scene].keys():
                dataDict[scene][restirScheme] = [profilerDict]
            else:
                dataDict[scene][restirScheme].append(profilerDict)

    exportDict = dict()
    for scene in dataDict.keys():
        exportDict[scene] = dict()

        for restirScheme in dataDict[scene].keys():
            print(f"Generating average profiler data for {scene}\\{restirScheme}")
            dictList = dataDict[scene][restirScheme]
            dictCount = len(dictList)
            minFrameCount = sys.maxsize

            for profilerDict in dictList:
                minFrameCount = min(minFrameCount, profilerDict["frameCount"])

            averageStats = {"min": 0.0, "max": 0.0, "mean": 0.0, "stdDev": 0.0}
            averageRecords = []

            for frame in range(minFrameCount):
                averageRecordValue = 0.0
    
                for profilerDict in dictList:
                    averageRecordValue += (1.0 / float(dictCount)) * profilerDict["events"]["/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/gpuTime"]["records"][frame]
    
                averageRecords.append(averageRecordValue)
    
            averageStats["min"] = min(averageRecords)
            averageStats["max"] = max(averageRecords)
            averageStats["mean"] = st.mean(averageRecords)
            averageStats["stdDev"] = st.stdev(averageRecords)
    
            exportDict[scene][restirScheme] = {"name": "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/gpuTime", "stats": averageStats, "records": averageRecords, "frameCount": minFrameCount}
        
    print("Exporting average profiler data", end="\r")
    with open(path.join(capturesPath, "profilerCapture.json"), "w") as file:
        file.write(json.dumps(exportDict))
        
    print("Exported average profiler data\t\n")


def naiveSchemeToGraphLabel(restirScheme):
    match restirScheme:
        case "Naive - ThreeQuarters":
            return "Naive - 3/4"
        case "Naive - Half":
            return "Naive - 1/2"
        case "Naive - Quarter":
            return "Naive - 1/4"
        case "Naive - OneEighth":
            return "Naive - 1/8"
        case "Naive - OneSixteenth":
            return "Naive - 1/16"
        case _:
            return restirScheme


def ExportGraphs():
    baseFolder = sys.argv[1]
    # ImageComparePath = sys.argv[2]

    print("Importing frame error and profiler data")
    capturesPath = path.join(baseFolder, "Captures")
    frameDataFilepath = path.join(capturesPath, "errors.json")
    profilerDataFilepath = path.join(capturesPath, "profilerCapture.json")

    frameDataDict = dict()
    profilerDataDict = dict()

    with open(frameDataFilepath, "r") as file:
        frameDataDict = json.load(file)

    with open(profilerDataFilepath, "r") as file:
        profilerDataDict = json.load(file)

    figuresPath = path.join(baseFolder, "Figures")
    if not path.exists(figuresPath):
        os.mkdir(figuresPath)

    # Create graphsDict
    scenes = set(frameDataDict.keys()).intersection(set(profilerDataDict.keys()))
    graphsDict = dict()
    for scene in scenes:
        graphsDict[scene] = dict()
        frameDataRestirSchemes = set(frameDataDict[scene].keys())
        profilerDataRestirSchemes = set(profilerDataDict[scene].keys())
        restirSchemes = frameDataRestirSchemes.intersection(profilerDataRestirSchemes)

        for restirScheme in restirSchemes:
            graphsDict[scene][restirScheme] = {"frametimes": (), "MAE": (), "MSE": (), "RMSE": (), "MAPE": ()}
            frameErrorDict = frameDataDict[scene][restirScheme]
            profilerDict = profilerDataDict[scene][restirScheme]

            errorPlotFrameCount = min(frameErrorDict["frameCount"], profilerDict["frameCount"])

            frameTimes = profilerDict["records"]
            cumulativeFrameTimes = []
            frameTimeSum = 0.0
            emptyFrameCount = 0
            for i in range(len(frameTimes)):
                frameTimeSum += frameTimes[i]
                if frameTimeSum == 0.0:
                    emptyFrameCount += 1
                    continue

                cumulativeFrameTimes.append(frameTimeSum)

            errorPlotFrameCount -= emptyFrameCount

            errorMAE = frameErrorDict["mae"][0:errorPlotFrameCount]
            errorMSE = frameErrorDict["mse"][0:errorPlotFrameCount]
            errorRMSE = frameErrorDict["rmse"][0:errorPlotFrameCount]
            errorMAPE = frameErrorDict["mape"][0:errorPlotFrameCount]

            errorFrameTimes = cumulativeFrameTimes[0:errorPlotFrameCount]
            graphsDict[scene][restirScheme]["frametimes"] = (cumulativeFrameTimes, frameTimes[emptyFrameCount:])
            graphsDict[scene][restirScheme]["MAE"] = (errorFrameTimes, errorMAE)
            graphsDict[scene][restirScheme]["MSE"] = (errorFrameTimes, errorMSE)
            graphsDict[scene][restirScheme]["RMSE"] = (errorFrameTimes, errorRMSE)
            graphsDict[scene][restirScheme]["MAPE"] = (errorFrameTimes, errorMAPE)

    # Export figures
    print("Generating graphs")
    for scene in graphsDict.keys():
        restirSchemes = sorted(graphsDict[scene].keys())

        # Frametime graph
        fig, ax = plt.subplots()

        for restirScheme in restirSchemes:
            cumulativeFrameTimes, frametimes = graphsDict[scene][restirScheme]["frametimes"]
            ax.plot(cumulativeFrameTimes, frametimes, label=naiveSchemeToGraphLabel(restirScheme))

        ax.set(xlabel="time (ms)", ylabel="frametime (ms)", title=f"frametimes ({scene})")
        ax.set_xbound(lower=0, upper=3000)
        ax.set_ybound(lower=0)
        ax.grid()
        ax.legend()
        fig.savefig(path.join(figuresPath, f"FrameTimes_{scene}.png"))
        print(f"Exported FrameTimes_{scene}.png")

        # Error Metrics
        errorMetrics = ["MAE", "MSE", "RMSE", "MAPE"]
        for errorMetric in errorMetrics:
            fig, ax = plt.subplots()

            for restirScheme in restirSchemes:
                cumulativeFrameTimes, errorValue = graphsDict[scene][restirScheme][errorMetric]
                ax.semilogy(cumulativeFrameTimes, errorValue, label=naiveSchemeToGraphLabel(restirScheme))

            ax.set(xlabel="time (ms)", ylabel=f"{errorMetric}", title=f"{errorMetric} ({scene})")
            ax.set_xbound(lower=0, upper=500)
            ax.set_ybound(lower=0)
            ax.grid()
            ax.legend()
            fig.savefig(path.join(figuresPath, f"{errorMetric}_{scene}.png"))
            print(f"Exported {errorMetric}_{scene}.png")

    print("Finished Exporting Graphs")


def main():
    ExportErrorMetrics()
    ExportAverageErrorMetrics()
    ExportAverageProfilerData()
    ExportGraphs()


if __name__ == "__main__":
    main()