from collections import defaultdict
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
import bs4
from bs4 import BeautifulSoup
import requests
session = requests.Session()
import os

# web = requests.get(url='http://khotruyen.com.vn/truyen-tranh/kinnikuman/chuong-1/54759.html').text
web = session.get(url='http://khotruyen.com.vn/truyen-tranh/kinnikuman').text
soup = BeautifulSoup(markup=web, features='lxml')
mangaName = 'lucsi-kinniku'
if not os.path.exists(mangaName):
   os.mkdir(mangaName)


for chapterAnchor in soup.find(name='table', class_='table-chapter').findAll(name="a"):
   chapterLink = chapterAnchor['href']
   chapterWebContent = session.get(url=chapterLink).text
   chapterSoup = BeautifulSoup(markup=chapterWebContent, features='lxml')
   print(f'working on {chapterLink}')

   for mediaItem in chapterSoup.findAll(name='div', class_='mediaItem'):
      imageLink = mediaItem.img['src']
      imageName = imageLink.split('/')[-1]
      imageNameSplited = imageName.split('-')
      imageNameSplited[0], imageNameSplited[1], imageNameSplited[2] = imageNameSplited[1], imageNameSplited[2], imageNameSplited[0]
      imageName = '-'.join(imageNameSplited)

      # image = requests.get(url=imageLink).content
      # print(f'writing {imageName}')

      with open(file=f'{mangaName}/{imageName}', mode='wb') as writeFile:
         image = session.get(url=imageLink).content
         print(f'writing {imageName}')
         writeFile.write(image)
