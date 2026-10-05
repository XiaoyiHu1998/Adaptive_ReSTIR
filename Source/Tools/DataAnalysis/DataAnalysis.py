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

            if run in runLogDict[scene].keys() and int(runLogDict[scene][run]) >= int(path.getmtime(runPath)):
                print(f"skipped comparing images in {runPath}\t")
                continue

            print(f"comparing images in {runPath}\t", end="\r")
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

            for frameIndex in range(len(runFrames)):
                runFrame = runFrames[frameIndex]
                referenceFrame = referenceFrames[frameIndex] if isAnimatedScene else referenceFrames[0]

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

        # Frametime graph naive schemes
        fig, ax = plt.subplots()
        for restirScheme in restirSchemes:
            excludedSchemes = ["TileBased", "PerPixel", "PerPixel - RISGuarantee"]

            if restirScheme not in excludedSchemes:
                cumulativeFrameTimes, frametimes = graphsDict[scene][restirScheme]["frametimes"]
                line, = ax.plot(cumulativeFrameTimes, frametimes, label=naiveSchemeToGraphLabel(restirScheme))

                # if "Naive" in restirScheme and not "Full" in restirScheme:
                #     line.set_dashes([4, 4])
                #     line.set_dash_capstyle("round")

        ax.set(xlabel="time (ms)", ylabel="frametime (ms)", title=f"Naive scheme frametimes ({scene})")
        ax.set_xbound(lower=0, upper=3000)
        ax.set_ybound(lower=0)
        ax.grid()
        ax.legend()
        figureName = f"NaiveSchemes_FrameTimes_{scene}.png"
        fig.savefig(path.join(figuresPath, figureName))
        print(f"Exported {figureName}")

        # Frametime graph main schemes
        fig, ax = plt.subplots()
        for restirScheme in restirSchemes:
            includedSchemes = ["TileBased", "PerPixel", "Naive - Full", "Naive - Quarter", "Naive - OneEighth"]

            if restirScheme in includedSchemes:
                cumulativeFrameTimes, frametimes = graphsDict[scene][restirScheme]["frametimes"]
                line, = ax.plot(cumulativeFrameTimes, frametimes, label=naiveSchemeToGraphLabel(restirScheme))

                if "Naive" in restirScheme and not "Full" in restirScheme:
                    line.set_dashes([4, 4])
                    line.set_dash_capstyle("round")

        ax.set(xlabel="time (ms)", ylabel="frametime (ms)", title=f"Main schemes frametimes ({scene})")
        ax.set_xbound(lower=0, upper=3000)
        ax.set_ybound(lower=0)
        ax.grid()
        ax.legend()
        figureName = f"MainSchemes_FrameTimes_{scene}.png"
        fig.savefig(path.join(figuresPath, figureName))
        print(f"Exported {figureName}")

        # Error Metrics Naive Scheme
        errorMetrics = ["MAE", "MSE", "RMSE", "MAPE"]
        for errorMetric in errorMetrics:
            fig, ax = plt.subplots()

            excludedSchemes = ["TileBased", "PerPixel", "PerPixel - RISGuarantee"]

            for restirScheme in restirSchemes:
                if restirScheme not in excludedSchemes:
                    cumulativeFrameTimes, errorValue = graphsDict[scene][restirScheme][errorMetric]
                    line, = ax.semilogy(cumulativeFrameTimes, errorValue, label=naiveSchemeToGraphLabel(restirScheme))

            ax.set(xlabel="time (ms)", ylabel=f"{errorMetric}", title=f"Naive schemes {errorMetric} ({scene})")
            ax.set_xbound(lower=0, upper=2000)
            ax.set_ybound(lower=0)
            ax.grid()
            ax.legend()
            figureName = f"NaiveSchemes_{errorMetric}_{scene}.png"
            fig.savefig(path.join(figuresPath, figureName))
            print(f"Exported {figureName}")

        # Error Metrics Main Schemes
        errorMetrics = ["MAE", "MSE", "RMSE", "MAPE"]
        for errorMetric in errorMetrics:
            fig, ax = plt.subplots()

            includedSchemes = ["TileBased", "PerPixel", "Naive - Full", "Naive - Quarter", "Naive - OneEighth"]

            for restirScheme in restirSchemes:
                if restirScheme in includedSchemes:
                    cumulativeFrameTimes, errorValue = graphsDict[scene][restirScheme][errorMetric]
                    line, = ax.semilogy(cumulativeFrameTimes, errorValue, label=naiveSchemeToGraphLabel(restirScheme))

                    if "Naive" in restirScheme and not "Full" in restirScheme:
                        line.set_dashes([4, 4])
                        line.set_dash_capstyle("round")

            ax.set(xlabel="time (ms)", ylabel=f"{errorMetric}", title=f"Main schemes {errorMetric} ({scene})")
            ax.set_xbound(lower=0, upper=2000)
            ax.set_ybound(lower=0)
            ax.grid()
            ax.legend()
            figureName = f"MainSchemes_{errorMetric}_{scene}.png"
            fig.savefig(path.join(figuresPath, figureName))
            print(f"Exported {figureName}")

    print("Finished Exporting Graphs")


def main():
    ExportErrorMetrics()
    ExportAverageErrorMetrics()
    ExportAverageProfilerData()
    ExportGraphs()


if __name__ == "__main__":
    main()