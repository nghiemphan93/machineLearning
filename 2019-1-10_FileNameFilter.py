import os, shutil

oneDrive          = "C:/Users/phan/OneDrive - adesso Group/Course"
oneDriveBooks  = "D:/OneDrive - adesso Group/Books"
compressed        = "C:/Users/phan/Downloads/Compressed"
folder            = oneDriveBooks

fileNames = os.listdir(folder)

for fileName in fileNames:
   # Clear the "." in file name && ignore file types
   '''
   fileTypes = [".pdf", ".azw3", ".epub", ".mobi"]
   filePath = os.path.join(folder, fileName)

   if "." in fileName:
      for fileType in fileTypes:
         if fileType in fileName:
            newFileName = fileName[:-len(fileType)]
            newFileName = newFileName.replace(".", " ")
            newFileName = newFileName + fileType
            newFilePath = os.path.join(folder, newFileName)

            if os.path.exists(newFilePath):
               print(newFilePath)
            else:
               os.rename(filePath, newFilePath)
   '''

   # remove "something" in the filename
   '''
   filePath = os.path.join(folder, fileName)
   toRemoveList = ["OReilly.", "Apress.", "CreateSpace.", "Manning.", "No.Starch.Press.Malware.", "Packt.",
                   "The.MIT.Press.", "Wiley.", "Martin.Hagan.", "[Bookflare.net] - "]
   for toRemove in toRemoveList:
      if toRemove in fileName:
         newFileName = fileName.replace(toRemove, "")
         newFilePath = os.path.join(folder, newFileName)
         if os.path.exists(newFilePath):
            print(newFilePath)
         else:
            os.rename(filePath, newFilePath)
   '''

   # Change to upper case && ignore file types

   fileTypes = [".pdf", ".azw3", ".epub", ".mobi"]
   filePath = os.path.join(folder, fileName)

   for fileType in fileTypes:
      if fileType in fileName:
         upperFileName = fileName[:-len(fileType)].upper()
         upperFileName = upperFileName + fileType
         upperFilePath = os.path.join(folder, upperFileName)

         os.rename(filePath, upperFilePath)





