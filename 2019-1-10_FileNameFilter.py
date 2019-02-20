import os, shutil

oneDrive          = "C:/Users/phan/OneDrive - adesso Group/Course"
oneDriveBooks     = "D:/OneDrive - adesso Group/Books"
sach              = "C:/Users/phan/Downloads/Sach"
compressed        = "C:/Users/phan/Downloads/Compressed"
books             = "C:/Users/Phan/Downloads/books"
newFolder         = "C:/Users/phan/Downloads/New folder"
folder            = books


# remove "something" in the filename
fileNames = os.listdir(folder)
for fileName in fileNames:
   filePath = os.path.join(folder, fileName)
   toRemoveList = ["OReilly.", "Apress.", "CreateSpace.", "Manning.", "No.Starch.Press.Malware.", "Packt.",
                   "The.MIT.Press.", "Wiley.", "Martin.Hagan.", "[Bookflare.net] - ", "The.MIT.Press.",
<<<<<<< HEAD
                   "Sachvui.Com-", "Packtpub - ", "Udemy - ", "[OREILLY] ", "[UdemyCourseDownloader] ", "_freetuts.download",
                   "[UDEMY] ", " - [FTU]", "[PEARSON] ", "[InFormIt] ", "[PACKT] ", "_[FREETUTS DOWNLOAD]", " - [FCO]", "[UDEMY] ", "_[freetuts.download]",
                   "_p30download.com", "Udemy.", "Pluralsight."]
=======
                   "Sachvui.Com-", "[smtebooks.eu] ", "Bản sao của ", "[smtebooks.eu] "]
>>>>>>> master
   for toRemove in toRemoveList:
      if toRemove in fileName:
         newFileName = fileName.replace(toRemove, "")
         newFilePath = os.path.join(folder, newFileName)
         if os.path.exists(newFilePath):
            print(newFilePath)
         else:
            os.rename(filePath, newFilePath)

fileNames = os.listdir(folder)
# Clear the "." and "-" in file name && ignore file types
for fileName in fileNames:
   fileTypes = [".pdf", ".azw3", ".epub", ".mobi", ".prc"]
   filePath = os.path.join(folder, fileName)

   if "." in fileName:
      for fileType in fileTypes:
         if fileType in fileName:
            newFileName = fileName[:-len(fileType)]
            newFileName = newFileName.replace(".", " ")
            newFileName = newFileName.replace("-", " ")
            newFileName = newFileName + fileType
            newFilePath = os.path.join(folder, newFileName)

            if os.path.exists(newFilePath):
               print(newFilePath)
            else:
               os.rename(filePath, newFilePath)

fileNames = os.listdir(folder)
# Change to upper case && ignore file types
for fileName in fileNames:
   fileTypes = [".pdf", ".azw3", ".epub", ".mobi", ".prc"]
   filePath = os.path.join(folder, fileName)

   if os.path.isdir(filePath):
      lowerFolderName = fileName
      lowerFolderPath = os.path.join(folder, lowerFolderName)
      upperFolderName = lowerFolderName.upper()
      upperFolderName = upperFolderName.replace(".", " ")
      upperFolderPath = os.path.join(folder, upperFolderName)
      os.rename(lowerFolderPath, upperFolderPath)
   else:
      for fileType in fileTypes:
         if fileType in fileName:
            upperFileName = fileName[:-len(fileType)].upper()
            upperFileName = upperFileName + fileType
            upperFilePath = os.path.join(folder, upperFileName)

            os.rename(filePath, upperFilePath)