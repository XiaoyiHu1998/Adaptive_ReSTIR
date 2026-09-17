import os
import os.path as path
import subprocess
import json
import sys

def generateErrorMetrics():
    baseFolder = sys.argv[1]
    ImageComparePath = sys.argv[2]

    frameCaptureFolderPath = path.join(baseFolder, "FrameCapture")
    subfolders = [subfolder for subfolder in os.listdir(frameCaptureFolderPath) if os.path.isdir(path.join(frameCaptureFolderPath, subfolder))]

    for subfolder in subfolders:
        print(f"comparing images in {subfolder}")
        splitSubfolderName = subfolder.split("_")
        # prefix = splitSubfolderName[0]
        scene = splitSubfolderName[-1]

        subFolderPath = path.join(frameCaptureFolderPath, subfolder)
        frames = [path.join(subFolderPath, frame) for frame in os.listdir(subFolderPath) if frame.split(".")[-1] == "png"]
        frames = sorted(frames)
        referenceFrame = path.join(frameCaptureFolderPath, "Reference_" + scene + ".png")

        errorsMAE = []
        errorsMSE = []
        errorsRMSE = []
        errorsMAPE = []

        print(f"current frame: {frames[0]}", end="\r")
        for frame in frames:
            print(f"current frame: {frame}", end="\r")

            maeCall = [ImageComparePath, "-m", "mae", frame, referenceFrame]
            mseCall = [ImageComparePath, "-m", "mse", frame, referenceFrame]
            rmseCall = [ImageComparePath, "-m", "rmse", frame, referenceFrame]
            mapeCall = [ImageComparePath, "-m", "mape", frame, referenceFrame]

            errorsMAE.append(subprocess.run(maeCall, capture_output=True, text=True).stdout.strip())
            errorsMSE.append(subprocess.run(mseCall, capture_output=True, text=True).stdout.strip())
            errorsRMSE.append(subprocess.run(rmseCall, capture_output=True, text=True).stdout.strip())
            errorsMAPE.append(subprocess.run(mapeCall, capture_output=True, text=True).stdout.strip())

        outputDict = dict()
        outputDict["mae"] = errorsMAE
        outputDict["mse"] = errorsMSE
        outputDict["rmse"] = errorsRMSE
        outputDict["mape"] = errorsMAPE

        with open(path.join(subFolderPath, "errors.json"), "w+") as file:
            file.write(json.dumps(outputDict))


def main():
    generateErrorMetrics()


if __name__ == "__main__":
    main()