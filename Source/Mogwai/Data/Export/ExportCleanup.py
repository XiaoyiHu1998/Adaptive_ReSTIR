import os
import os.path as path
import Data.Export.ExportConfig as cfg


def cleanFolder(folderPath):
    print(f"Cleaning up .exr files from {folderPath}", end="\r")
    
    subFolders = [subFolder for subFolder in os.listdir(folderPath) if path.isdir(path.join(folderPath, subFolder))]
    files = [file for file in os.listdir(folderPath) if path.isfile(path.join(folderPath, file))]

    for file in files:
        if file.split(".")[-1] == "exr":
            filePath = path.join(folderPath, file)
            os.remove(filePath)

    for subFolder in subFolders:
        subFolderPath = path.join(folderPath, subFolder)
        print(subFolderPath)
        cleanFolder(subFolderPath)


def main():
    print(f"Cleaning up .exr files from {capturesFolder} and its subfolders")

    capturesFolder = path.join(cfg.baseDirectory, "Captures")
    cleanFolder(capturesFolder)

    print(f"Finished cleaning up .exr files from {capturesFolder} and its subfolders\n")


if __name__ == "__main__":
    main()