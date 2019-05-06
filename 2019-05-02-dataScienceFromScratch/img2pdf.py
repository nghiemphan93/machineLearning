# region Import
from collections import defaultdict
import os
from fpdf import FPDF
from PIL import Image


# endregion



def makePdf(pdfFileName, imageNames, folder='./', subFolder='pdf'):
   outputFolder = os.path.join(folder, subFolder)
   if os.path.exists(outputFolder) is False:
      os.mkdir(outputFolder)

   # cover = Image.open(os.path.join(folder, imageNames[0]))
   # width, height = cover.size
   # 735x1200 applies for just this manga
   pdf = FPDF(unit="pt", format=[735, 1200])
   for index, image in enumerate(imageNames):
      pdf.add_page()
      imagePath = os.path.join(folder, image)
      pdf.image(imagePath, 0, 0)

   outputPdfPath = f'{os.path.join(outputFolder, pdfFileName)}.pdf'

   pdf.output(name=outputPdfPath, dest='F')


# folder = 'C:/Users/phan/OneDrive - adesso Group/Truyen Tranh/lucsi-kinniku'
folder = 'C:/Users/phan/Downloads/Sach/lucsi-kinniku'
pdf = FPDF()

chapter2files = defaultdict(list)
allFiles = [file for file in os.listdir(folder)
            if os.path.isdir(file) is False]
allFiles.sort(key=lambda fileName: (int(fileName.split('-')[1]), int(fileName.split('-')[2])))

for fileName in allFiles:
   fileNameSplited = fileName.split('-')
   chapterName = f'{fileNameSplited[0]}-{fileNameSplited[1]}'
   chapter2files[chapterName].append(fileName)

counter = 0
for chapter, files in chapter2files.items():
   print(f'working on {chapter} {counter / len(chapter2files.items()) * 100:.2f}% completed')
   makePdf(pdfFileName=chapter, imageNames=files, folder=folder)
   counter += 1

