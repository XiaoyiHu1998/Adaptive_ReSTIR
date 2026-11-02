import os
import os.path as path
import subprocess
import json
import sys
import shutil
import matplotlib.pyplot as plt
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import matplotlib.patches as patches


def ExecuteImageCompareCall(commandList):
    return subprocess.run(commandList, capture_output=True, text=True).stdout.strip()


def ExecuteFlipImageCompareCall(commandList):
    output = subprocess.run(commandList, capture_output=True, text=True).stdout
    output = output.split("\n")[1].strip().split(" ")[-1]
    return output


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
            flipDirPath = path.join(runPath, "Flip")

            if not path.exists(flipDirPath):
                os.makedirs(flipDirPath)

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
            errorsFLIP = []
            
            maeCalls = []
            mseCalls = []
            rmseCalls = []
            mapeCalls = []
            flipCalls = []

            frameComparisonCount = min(len(runFrames), len(referenceFrames)) if isAnimatedScene else len(runFrames)
            for frameIndex in range(frameComparisonCount):
                runFrame = runFrames[frameIndex]
                referenceFrame = referenceFrames[frameIndex] if isAnimatedScene else referenceFrames[0]

                maeCall = [ImageComparePath, "-m", "mae", runFrame, referenceFrame]
                mseCall = [ImageComparePath, "-m", "mse", runFrame, referenceFrame]
                rmseCall = [ImageComparePath, "-m", "rmse", runFrame, referenceFrame]
                mapeCall = [ImageComparePath, "-m", "mape", runFrame, referenceFrame]
                flipCall = ["flip", "-r", referenceFrame, "-t", runFrame, "-d", flipDirPath, "-v", "1"]

                maeCalls.append(maeCall)
                mseCalls.append(mseCall)
                rmseCalls.append(rmseCall)
                mapeCalls.append(mapeCall)
                flipCalls.append(flipCall)

            with ThreadPoolExecutor(max_workers=16) as executor:
                errorsMAE = list(executor.map(ExecuteImageCompareCall, maeCalls))
                errorsMSE = list(executor.map(ExecuteImageCompareCall, mseCalls))
                errorsRMSE = list(executor.map(ExecuteImageCompareCall, rmseCalls))
                errorsMAPE = list(executor.map(ExecuteImageCompareCall, mapeCalls))
                errorsFLIP = list(executor.map(ExecuteFlipImageCompareCall, flipCalls))

            exportDict = dict()
            exportDict["frameCount"] = frameComparisonCount
            exportDict["mae"] = errorsMAE
            exportDict["mse"] = errorsMSE
            exportDict["rmse"] = errorsRMSE
            exportDict["mape"] = errorsMAPE
            exportDict["flip"] = errorsFLIP

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
                print(f"minFrameCount: {minFrameCount}", end="\n")
    
            averageMAE = []
            averageMSE = []
            averageRMSE = []
            averageMAPE = []
            averageFLIP = []

            for frameIndex in range(minFrameCount):
                mae = 0.0
                mse = 0.0
                rmse = 0.0
                mape = 0.0
                flip = 0.0

                for dictIndex in range(dictCount):
                    mae += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mae"][frameIndex])
                    mse += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mse"][frameIndex])
                    rmse += (1.0 / float(dictCount)) * float(dictList[dictIndex]["rmse"][frameIndex])
                    mape += (1.0 / float(dictCount)) * float(dictList[dictIndex]["mape"][frameIndex])
                    flip += (1.0 / float(dictCount)) * float(dictList[dictIndex]["flip"][frameIndex])

                floorNoise = lambda averageError : 0.0 if averageError < 0.0001 else averageError
    
                averageMAE.append(floorNoise(mae))
                averageMSE.append(floorNoise(mse))
                averageRMSE.append(floorNoise(rmse))
                averageMAPE.append(floorNoise(mape))
                averageFLIP.append(floorNoise(flip))

            averageErrorDict = dict()
            averageErrorDict["frameCount"] = minFrameCount
            averageErrorDict["mae"] = averageMAE
            averageErrorDict["mse"] = averageMSE
            averageErrorDict["rmse"] = averageRMSE
            averageErrorDict["mape"] = averageMAPE
            averageErrorDict["flip"] = averageFLIP

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

    # profilerDict keys for different passes
    ReSTIRPTPass = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/gpuTime"
    generatePaths = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/generatePaths/gpuTime"
    tracePass = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/tracePass/gpuTime"
    # temporalPathRetrace = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/temporalPathRetrace/gpuTime"
    # temporalReuse = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/temporalReuse/gpuTime"
    spatialPathRetrace = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/spatialPathRetrace/gpuTime"
    spatialReuse = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/spatialReuse/gpuTime"
    EmissivePowerSamplerUpdate = "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/EmissivePowerSampler::update/gpuTime"

    generatePathsPassName = lambda restirScheme: "generatePathsNaive" if "Naive" in restirScheme else ("generatePathsPerPixel" if "PerPixel" in restirScheme else "generatePathsTileBased")

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

            ReSTIRPTStats = {"min": 0.0, "max": 0.0, "mean": 0.0, "stdDev": 0.0}
            averageReSTIRPTPass = []
            averageGeneratePaths = []
            averageTracePass = []
            averageSpatialPathRetrace = []
            averageSpatialReuse = []
            averageEmissivePowerSamplerUpdate = []

            generatePathsScheme = generatePaths.replace("generatePaths", generatePathsPassName(restirScheme))

            for frame in range(minFrameCount):
                averageReSTIRPTPassValue = 0.0
                averageGeneratePathsValue = 0.0
                averageTracePassValue = 0.0
                averageSpatialPathRetraceValue = 0.0
                averageSpatialReuseValue = 0.0
                averageEmissivePowerSamplerUpdateValue = 0.0
    
                for profilerDict in dictList:
                    averageReSTIRPTPassValue += (1.0 / float(dictCount)) * profilerDict["events"][ReSTIRPTPass]["records"][frame]
                    averageGeneratePathsValue += (1.0 / float(dictCount)) * profilerDict["events"][generatePathsScheme]["records"][frame]
                    averageTracePassValue += (1.0 / float(dictCount)) * profilerDict["events"][tracePass]["records"][frame]
                    averageSpatialPathRetraceValue += (1.0 / float(dictCount)) * profilerDict["events"][spatialPathRetrace]["records"][frame]
                    averageSpatialReuseValue += (1.0 / float(dictCount)) * profilerDict["events"][spatialReuse]["records"][frame]
                    averageEmissivePowerSamplerUpdateValue += (1.0 / float(dictCount)) * profilerDict["events"][EmissivePowerSamplerUpdate]["records"][frame]
    
                averageReSTIRPTPass.append(averageReSTIRPTPassValue)
                averageGeneratePaths.append(averageGeneratePathsValue)
                averageTracePass.append(averageTracePassValue)
                averageSpatialPathRetrace.append(averageSpatialPathRetraceValue)
                averageSpatialReuse.append(averageSpatialReuseValue)
                averageEmissivePowerSamplerUpdate.append(averageEmissivePowerSamplerUpdateValue)
    
            ReSTIRPTStats["min"] = min(averageReSTIRPTPass)
            ReSTIRPTStats["max"] = max(averageReSTIRPTPass)
            ReSTIRPTStats["mean"] = st.mean(averageReSTIRPTPass)
            ReSTIRPTStats["stdDev"] = st.stdev(averageReSTIRPTPass)
    
            exportDict[scene][restirScheme] = {
                "name": "/onFrameRender/RenderGraphExe::execute()/ReSTIRPTPass/gpuTime",
                "ReSTIRPTStats": ReSTIRPTStats,
                "ReSTIRPTPass": averageReSTIRPTPass,
                "generatePaths": averageGeneratePaths,
                "tracePass": averageTracePass,
                "spatialPathRetrace": averageSpatialPathRetrace,
                "spatialReuse": averageSpatialReuse,
                "emissivePowerSamplerUpdate": averageEmissivePowerSamplerUpdate,
                "frameCount": minFrameCount
                }
        
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


def GenerateGraphsDict():
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

    # Create graphsDict
    scenes = set(frameDataDict.keys()).intersection(set(profilerDataDict.keys()))
    graphsDict = dict()
    for scene in scenes:
        graphsDict[scene] = dict()
        frameDataRestirSchemes = set(frameDataDict[scene].keys())
        profilerDataRestirSchemes = set(profilerDataDict[scene].keys())
        restirSchemes = frameDataRestirSchemes.intersection(profilerDataRestirSchemes)

        for restirScheme in restirSchemes:
            graphsDict[scene][restirScheme] = {"frametimes": (), "MAE": (), "MSE": (), "RMSE": (), "MAPE": (), "FLIP": ()}
            frameErrorDict = frameDataDict[scene][restirScheme]
            profilerDict = profilerDataDict[scene][restirScheme]

            frameTimes = profilerDict["ReSTIRPTPass"]
            frameTimes = [frameTime for frameTime in frameTimes if frameTime > 0.0 and frameTime < 500.0]
            frameCount = len(frameTimes)
            frametimesReSTIROverhead = [total - tracePass for (total, tracePass) in zip(profilerDict["ReSTIRPTPass"], profilerDict["tracePass"]) if total > 0.0 and total < 500.0]
            cumulativeFrameTimes = []
            frameTimeSum = 0.0

            for i in range(len(frameTimes)):
                frameTimeSum += frameTimes[i]
                cumulativeFrameTimes.append(frameTimeSum)

            errorPlotFrameCount = min(frameErrorDict["frameCount"], profilerDict["frameCount"])
            errorPlotFrameCount = min(errorPlotFrameCount, frameCount)

            errorMAE = frameErrorDict["mae"][0:errorPlotFrameCount]
            errorMSE = frameErrorDict["mse"][0:errorPlotFrameCount]
            errorRMSE = frameErrorDict["rmse"][0:errorPlotFrameCount]
            errorMAPE = frameErrorDict["mape"][0:errorPlotFrameCount]
            errorFLIP = frameErrorDict["flip"][0:errorPlotFrameCount]

            errorFrameTimes = cumulativeFrameTimes[0:errorPlotFrameCount]
            graphsDict[scene][restirScheme]["frametimes"] = (cumulativeFrameTimes, frameTimes)
            graphsDict[scene][restirScheme]["MAE"] = (errorFrameTimes, errorMAE)
            graphsDict[scene][restirScheme]["MSE"] = (errorFrameTimes, errorMSE)
            graphsDict[scene][restirScheme]["RMSE"] = (errorFrameTimes, errorRMSE)
            graphsDict[scene][restirScheme]["MAPE"] = (errorFrameTimes, errorMAPE)
            graphsDict[scene][restirScheme]["Mean FLIP"] = (errorFrameTimes, errorFLIP)

            if restirScheme == "Naive - OneSixteenth":
                graphsDict[scene][restirScheme]["frametimesReSTIROverhead"] = (cumulativeFrameTimes, frametimesReSTIROverhead)

    return graphsDict


def ExportGraphs():
    baseFolder = sys.argv[1]
    # ImageComparePath = sys.argv[2]

    graphsDict = GenerateGraphsDict()

    figuresPath = path.join(path.join(baseFolder, "Figures"), "Graphs")
    if not path.exists(figuresPath):
        os.makedirs(figuresPath)

    print("Generating graphs")
    for scene in graphsDict.keys():
        restirSchemes = sorted(graphsDict[scene].keys())
        animatedScene = "Animated" in scene

        # Frametime graph naive schemes
        fig, ax = plt.subplots()
        for restirScheme in restirSchemes:
            excludedSchemes = ["TileBased", "PerPixel", "PerPixel - RISGuarantee"]

            if restirScheme not in excludedSchemes:
                cumulativeFrameTimes, frametimes = graphsDict[scene][restirScheme]["frametimes"]
                line, = ax.plot(cumulativeFrameTimes, frametimes, label=naiveSchemeToGraphLabel(restirScheme))

                if restirScheme == "Naive - OneSixteenth":
                    cumulativeFrameTimes, frametimesReSTIROverhead = graphsDict[scene][restirScheme]["frametimesReSTIROverhead"]
                    line, = ax.plot(cumulativeFrameTimes, frametimesReSTIROverhead, label="Naive - 1/16 No Rays")
                    line.set_dashes([4, 4])
                    line.set_dash_capstyle("round")

                # if "Naive" in restirScheme and not "Full" in restirScheme:
                #     line.set_dashes([4, 4])
                #     line.set_dash_capstyle("round")

        ax.set(xlabel="time (ms)", ylabel="frametime (ms)", title=f"Naive scheme frametimes ({scene})")
        ax.set_xbound(lower=0, upper=3000)
        ax.set_ybound(lower=0)
        ax.grid()
        ax.legend()
        figureName = f"NaiveSchemes_FrameTimes_{scene}.pdf"
        fig.savefig(path.join(figuresPath, figureName), bbox_inches="tight", format="pdf")
        print(f"Exported {figureName}")

        # Frametime graph main schemes
        fig, ax = plt.subplots()
        for restirScheme in restirSchemes:
            includedSchemes = ["TileBased", "PerPixel", "Naive - Full", "Naive - Quarter", "Naive - OneEighth", "Naive - OneSixteenth"]

            if restirScheme in includedSchemes:
                if restirScheme !="Naive - OneSixteenth":
                    cumulativeFrameTimes, frametimes = graphsDict[scene][restirScheme]["frametimes"]
                    line, = ax.plot(cumulativeFrameTimes, frametimes, label=naiveSchemeToGraphLabel(restirScheme))

                if restirScheme == "Naive - OneSixteenth":
                    cumulativeFrameTimes, frametimesReSTIROverhead = graphsDict[scene][restirScheme]["frametimesReSTIROverhead"]
                    line, = ax.plot(cumulativeFrameTimes, frametimesReSTIROverhead, label="Naive - 1/16 No Rays")
                    line.set_dashes([4, 4])
                    line.set_dash_capstyle("round")

                if "Naive" in restirScheme and not "Full" in restirScheme:
                    line.set_dashes([4, 4])
                    line.set_dash_capstyle("round")

        ax.set(xlabel="time (ms)", ylabel="frametime (ms)", title=f"Main schemes frametimes ({scene})")
        ax.set_xbound(lower=0, upper=3000)
        ax.set_ybound(lower=0)
        ax.grid()
        ax.legend()
        figureName = f"MainSchemes_FrameTimes_{scene}.pdf"
        fig.savefig(path.join(figuresPath, figureName), bbox_inches="tight", format="pdf")
        print(f"Exported {figureName}")

        # Error Metrics Naive Scheme
        errorMetrics = ["MAE", "MSE", "RMSE", "MAPE", "Mean FLIP"]
        for errorMetric in errorMetrics:
            fig, ax = plt.subplots()

            excludedSchemes = ["TileBased", "PerPixel", "PerPixel - RISGuarantee"]

            upperBoundX = sys.maxsize
            for restirScheme in restirSchemes:
                if restirScheme not in excludedSchemes:
                    cumulativeFrameTimes, errorValue = graphsDict[scene][restirScheme][errorMetric]
                    xValues = cumulativeFrameTimes

                    if animatedScene:
                        xValues = list(range(len(errorValue)))

                    if(len(xValues) != len(errorValue)):
                        print(f"ERROR Inconsistent X and Y Axis Value Count: MainSchemes_{errorMetric}_{scene}.pdf")
                        continue

                    upperBoundX = min(upperBoundX, xValues[-1])
                    line, = ax.semilogy(xValues, errorValue, label=naiveSchemeToGraphLabel(restirScheme))

            xLabel = "Frame" if animatedScene else "time (ms)"
            ax.set(xlabel=xLabel, ylabel=f"{errorMetric}", title=f"Naive schemes {errorMetric} ({scene})")
            ax.set_xbound(lower=0, upper=upperBoundX)
            ax.set_ybound(lower=0)
            ax.grid()
            ax.grid(which="minor", color="0.9")
            ax.legend()
            figureName = f"NaiveSchemes_{errorMetric}_{scene}.pdf"
            fig.savefig(path.join(figuresPath, figureName), bbox_inches="tight", format="pdf")
            print(f"Exported {figureName}")

        # Error Metrics Main Schemes
        errorMetrics = ["MAE", "MSE", "RMSE", "MAPE", "Mean FLIP"]
        for errorMetric in errorMetrics:
            fig, ax = plt.subplots()

            includedSchemes = ["TileBased", "PerPixel", "Naive - Full", "Naive - Quarter", "Naive - OneEighth"]

            upperBoundX = sys.maxsize
            for restirScheme in restirSchemes:
                if restirScheme in includedSchemes:
                    cumulativeFrameTimes, errorValue = graphsDict[scene][restirScheme][errorMetric]
                    xValues = cumulativeFrameTimes

                    if animatedScene:
                        xValues = list(range(len(errorValue)))

                    if(len(xValues) != len(errorValue)):
                        print(f"ERROR Inconsistent X and Y Axis Value Count: MainSchemes_{errorMetric}_{scene}.pdf")
                        continue

                    upperBoundX = min(upperBoundX, xValues[-1])
                    line, = ax.semilogy(xValues, errorValue, label=naiveSchemeToGraphLabel(restirScheme))

                    if "Naive" in restirScheme and not "Full" in restirScheme:
                        line.set_dashes([4, 4])
                        line.set_dash_capstyle("round")

            xLabel = "Frame" if animatedScene else "time (ms)"
            ax.set(xlabel=xLabel, ylabel=f"{errorMetric}", title=f"Main schemes {errorMetric} ({scene})")
            ax.set_xbound(lower=0, upper=upperBoundX)
            ax.set_ybound(lower=0)
            ax.grid()
            ax.grid(which="minor", color="0.9")
            ax.legend()
            figureName = f"MainSchemes_{errorMetric}_{scene}.pdf"
            fig.savefig(path.join(figuresPath, figureName), bbox_inches="tight", format="pdf")
            print(f"Exported {figureName}")

    print("Finished Exporting Graphs\n")


def ExportTableData():
    baseFolder = sys.argv[1]
    # ImageComparePath = sys.argv[2]

    capturesPath = path.join(baseFolder, "Captures")
    profilerDictPath = path.join(capturesPath, "profilerCapture.json")

    if not path.exists(profilerDictPath):
        return

    profilerCaptureDict = dict()
    with open(profilerDictPath, "r") as file:
        profilerCaptureDict = json.load(file)

    exportDict = dict()
    for scene in profilerCaptureDict.keys():
        exportDict[scene] = dict()
        for restirScheme in profilerCaptureDict[scene].keys():
            exportDict[scene][restirScheme] = dict()
            profilerDict = profilerCaptureDict[scene][restirScheme]

            ReSTIRPTPassTimes = profilerDict["ReSTIRPTPass"]
            generatePathsTimes = profilerDict["generatePaths"]
            tracePassTimes = profilerDict["tracePass"]
            spatialPathRetraceTimes = profilerDict["spatialPathRetrace"]
            spatialReuseTimes = profilerDict["spatialReuse"]
            emissivePowerSamplerUpdateTimes = profilerDict["emissivePowerSamplerUpdate"]

            nonTemporalTimes = list(map(sum, zip(generatePathsTimes, tracePassTimes, spatialPathRetraceTimes, spatialReuseTimes, emissivePowerSamplerUpdateTimes)))
            temporalReusePassTimes = [total - nonTemporal for (total, nonTemporal) in zip(ReSTIRPTPassTimes, nonTemporalTimes)]
            spatialReusePassTimes = list(map(sum, zip(spatialPathRetraceTimes, spatialReuseTimes)))

            startIndex = -51
            endIndex = -1
            exportDict[scene][restirScheme]["ReSTIRPT_Total"] = round(sum(ReSTIRPTPassTimes[startIndex:endIndex]) / float(len(ReSTIRPTPassTimes[startIndex:endIndex])), 2)
            exportDict[scene][restirScheme]["generatePaths"] = round(sum(generatePathsTimes[startIndex:endIndex]) / float(len(generatePathsTimes[startIndex:endIndex])), 2)
            exportDict[scene][restirScheme]["tracePaths"] = round(sum(tracePassTimes[startIndex:endIndex]) / float(len(tracePassTimes[startIndex:endIndex])), 2)
            exportDict[scene][restirScheme]["temporalReuse"] = round(sum(temporalReusePassTimes[startIndex:endIndex]) / float(len(temporalReusePassTimes[startIndex:endIndex])), 2)
            exportDict[scene][restirScheme]["spatialReuse"] = round(sum(spatialReusePassTimes[startIndex:endIndex]) / float(len(spatialReusePassTimes[startIndex:endIndex])), 2)
            exportDict[scene][restirScheme]["emissivePowerSamplerUpdate"] = round(sum(emissivePowerSamplerUpdateTimes[startIndex:endIndex]) / float(len(emissivePowerSamplerUpdateTimes[startIndex:endIndex])), 2)

    exportDictName = "performancePerPass.json"
    with open(path.join(capturesPath, exportDictName), "w+") as file:
        file.write(json.dumps(exportDict, indent=4))
        
    print(f"Exported {exportDictName}")


def GatherComparisonData():
    baseFolder = sys.argv[1]
    capturesFolder = path.join(baseFolder, "Captures")
    comparisonDataFolder = path.join(baseFolder, "ComparisonData")

    print(f"Gathering comparison data in {comparisonDataFolder}")

    captureScenes = [subfolder for subfolder in os.listdir(capturesFolder) if path.isdir(path.join(capturesFolder, subfolder))]
    for captureScene in captureScenes:
        sceneFolder = path.join(comparisonDataFolder, captureScene)
        if not path.exists(sceneFolder):
            os.mkdir(sceneFolder)

        captureSceneFolder = path.join(capturesFolder, captureScene)
        schemes = [subfolder for subfolder in os.listdir(captureSceneFolder) if subfolder.split("_")[-1] == "1" or subfolder == "Reference"]
        for scheme in schemes:
            print(f"Gathering comparison data from {captureScene}\\{scheme}")
            schemeFolder = path.join(path.join(capturesFolder, captureScene), scheme)
            frames = sorted([file for file in os.listdir(schemeFolder) if path.isfile(path.join(schemeFolder, file)) and file.split(".")[-1] == "png"])
            lastFrame= frames[-1]
            lastFramePath = path.join(schemeFolder, lastFrame)

            schemeName = scheme.split("_")[0]
            comparisonSchemeFolder = path.join(path.join(comparisonDataFolder, captureScene), schemeName)
            if not path.exists(comparisonSchemeFolder):
                os.makedirs(comparisonSchemeFolder)
            shutil.copy2(lastFramePath, path.join(comparisonSchemeFolder, lastFrame))
            
            renamedFile = path.join(comparisonSchemeFolder, f"{captureScene}_{schemeName}.png")
            if path.exists(renamedFile):
                os.remove(renamedFile)
            os.rename(path.join(comparisonSchemeFolder, lastFrame), renamedFile)

            if scheme != "Reference":
                flipFolder = path.join(schemeFolder, "Flip")
                flipFrames =  sorted([file for file in os.listdir(flipFolder) if path.isfile(path.join(flipFolder, file)) and file.split(".")[-1] == "png"])
                lastFlipFrame = flipFrames[-1]
                lastFlipFramePath = path.join(flipFolder, lastFlipFrame)
                comparisonFlipFolder = path.join(comparisonSchemeFolder, "Flip")
                if not path.exists(comparisonFlipFolder):
                    os.makedirs(comparisonFlipFolder)
                shutil.copy2(lastFlipFramePath, path.join(comparisonFlipFolder, lastFlipFrame))

                renamedFlipFile = path.join(comparisonFlipFolder, f"{captureScene}_{schemeName}_FLIP.png")
                if path.exists(renamedFlipFile):
                    os.remove(renamedFlipFile)
                os.rename(path.join(comparisonFlipFolder, lastFlipFrame), renamedFlipFile)

    comparisonScenes = [subfolder for subfolder in os.listdir(comparisonDataFolder) if path.isdir(path.join(comparisonDataFolder, subfolder))]
    for comparisonScene in comparisonScenes:
        if comparisonScene not in captureScenes:
            os.removedirs(path.join(comparisonDataFolder, comparisonScene))
    
    print(f"Finished gathering comparison data in {comparisonDataFolder}\n")


def CreateComparisonFigures():
    baseFolder = sys.argv[1]
    exportFolder = path.join(path.join(baseFolder, "Figures"), "Comparisons")
    comparisonDataFolder = path.join(baseFolder, "ComparisonData")

    if not path.exists(exportFolder):
        os.makedirs(exportFolder)

    scenes = [subfolder for subfolder in os.listdir(comparisonDataFolder) if path.isdir(path.join(comparisonDataFolder, subfolder))]
    imagesDict = {}

    schemeFontSize = 14
    regionFontSize = 16

    comparisonsDict = dict()
    comparisonsJsonPath = path.join(comparisonDataFolder, "comparisons.json")
    if not path.exists(comparisonsJsonPath):
        return

    with open(comparisonsJsonPath, "r") as file:
        comparisonsDict = json.load(file)

    print("Generating Comparison Figures")

    for scene in scenes:
        sceneFolder = path.join(comparisonDataFolder, scene)
        schemes = [subfolder for subfolder in os.listdir(sceneFolder) if path.isdir(path.join(sceneFolder, subfolder))]
        imagesDict[scene] = dict()

        for scheme in schemes:
            schemeFolder = path.join(sceneFolder, scheme)
            imageName = [file for file in os.listdir(schemeFolder) if path.isfile(path.join(schemeFolder, file))][0]
            imagesDict[scene][scheme] = dict()
            imagesDict[scene][scheme]["Frame"] = Image.open(path.join(schemeFolder, imageName)).convert("RGB")

            if scheme != "Reference":
                flipImageName = imageName.split(".")[0] + "_FLIP.png"
                imagesDict[scene][scheme]["Flip"] = Image.open(path.join(path.join(schemeFolder, "Flip"), flipImageName)).convert("RGB")

        if scene not in comparisonsDict.keys():
            continue
            
        for comparison in comparisonsDict[scene].keys():
            regionCount = len(comparisonsDict[scene][comparison].keys())
            heightRatios = [1 for regionKey in comparisonsDict[scene][comparison].keys()]
            schemeCount = len(schemes)

            figure = plt.figure(figsize=[1920, 1080, "px"])
            gs = figure.add_gridspec(1, 1, height_ratios=[1], hspace=0, wspace=0)
            ax_main = figure.add_subplot(gs[0, :])
            ax_main.imshow(imagesDict[scene]["Reference"]["Frame"])
            ax_main.axis("off")
            ax_main.margins(0)
            
            boxColors = ["red", "green", "blue", "yellow"]
            for (region, color) in zip(comparisonsDict[scene][comparison].keys(), boxColors):
                xStart, xEnd, yStart, yEnd = tuple(comparisonsDict[scene][comparison][region])
                regionBox = patches.Rectangle((xStart, yStart), xEnd - xStart, yEnd - yStart, linewidth=5, edgecolor=color, facecolor="none")
                ax_main.add_patch(regionBox)

            figureName = f"{scene}_{comparison}_Top.pdf"
            plt.savefig(path.join(exportFolder, figureName), bbox_inches="tight", dpi="figure", pad_inches=0, format="pdf")
            print(f"Generated {figureName}")

            comparisonCount = len(comparisonsDict[scene][comparison].items())
            figHeight = 347 * comparisonCount
            figure = plt.figure(figsize=[1548, figHeight, "px"])
            gs = figure.add_gridspec(regionCount * 2, schemeCount, height_ratios=heightRatios + heightRatios, hspace=0, wspace=0)
            for row, (region, regionCoords) in enumerate(comparisonsDict[scene][comparison].items(), start=0):
                row = row * 2
                cropCoords = (regionCoords[0], regionCoords[2], regionCoords[1], regionCoords[3])

                schemeOrder = ["Naive - OneSixteenth", "Naive - OneEighth", "Naive - Quarter", "Naive - Half", "Naive - ThreeQuarters", "Naive - Full", "PerPixel", "TileBased", "Reference"]
                schemeNames = ["Naive - 1/16", "Naive - 1/8", "Naive - 1/4", "Naive - 1/2", "Naive - 3/4", "Naive - Full", "PerPixel", "TileBased", "Reference"]

                schemeOrder = list(reversed(schemeOrder))
                schemeNames = list(reversed(schemeNames))

                for i in range(len(schemeOrder)):
                    image = imagesDict[scene][schemeOrder[i]]["Frame"]
                    croppedImage = image.crop(cropCoords)

                    ax = figure.add_subplot(gs[row, i])
                    ax.imshow(croppedImage)
                    ax.margins(0)
                    ax.axis("off")

                    if row == 0:
                        ax.set_title(schemeNames[i], fontsize=schemeFontSize)

                    if i == 0 and row % 2 == 0:
                        ax.text(-0.05, 0.5, region, transform=ax.transAxes, va="center", ha="right", fontsize=regionFontSize)
                    

                row += 1
                for i in range(len(schemeOrder)):
                    if schemeOrder[i] == "Reference":
                        continue

                    image = imagesDict[scene][schemeOrder[i]]["Flip"]
                    croppedImage = image.crop(cropCoords)

                    ax = figure.add_subplot(gs[row, i])
                    ax.imshow(croppedImage)
                    ax.margins(0)
                    ax.axis("off")

            figureName = f"{scene}_{comparison}_Bottom.pdf"
            plt.savefig(path.join(exportFolder, figureName), bbox_inches="tight", dpi=300, pad_inches=0, format="pdf")
            print(f"Generated {figureName}")

    print("Finished Generating Comparison Figures\n")
    

def main():
    ExportErrorMetrics()
    ExportAverageErrorMetrics()
    ExportAverageProfilerData()
    ExportGraphs()
    ExportTableData()
    GatherComparisonData()
    CreateComparisonFigures()


if __name__ == "__main__":
    main()