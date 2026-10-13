import os
import os.path as path
import ExportConfig as cfg


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
        cleanFolder(subFolderPath)


def main():
    capturesFolder = path.join(cfg.baseDirectory, "Captures")
    print(f"Cleaning up .exr files from {capturesFolder} and its subfolders")

    cleanFolder(capturesFolder)

    print(f"Finished cleaning up .exr files from {capturesFolder} and its subfolders\t\n")


if __name__ == "__main__":
    main()