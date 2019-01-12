import os, shutil

oneDrive        = "C:/Users/phan/OneDrive - adesso Group/Course"
compressed  = "C:/Users/phan/Downloads/Compressed"
folder      = oneDrive

fileNames = os.listdir(folder)

for fileName in fileNames:
   # Clear the "." in file name

   filePath = os.path.join(folder, fileName)
   if "." in fileName:
      newFileName = fileName.replace(".", " ")
      newFilePath = os.path.join(folder, newFileName)

      if os.path.exists(newFilePath):
         print(newFilePath)
      else:
         os.rename(filePath, newFilePath)


   # remove "something" in the filename
   '''
   filePath = os.path.join(folder, fileName)
   toRemove = "OReilly."
   if toRemove in fileName:
      newFileName = fileName.replace(toRemove, "")
      newFilePath = os.path.join(folder, newFileName)

      if os.path.exists(newFilePath):
         print(newFilePath)
      else:
         os.rename(filePath, newFilePath)
   '''

   # Change to upper case
   '''
   filePath = os.path.join(folder, fileName)
   upperFileName = fileName.upper()
   upperFilePath = os.path.join(folder, upperFileName)

   print(upperFilePath)
   
   os.rename(filePath, upperFilePath)
   '''



