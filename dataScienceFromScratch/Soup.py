from collections import defaultdict
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import bs4
from bs4 import BeautifulSoup
import requests

web = requests.get(url='http://coreyms.com').text
soup = BeautifulSoup(markup=web, features='lxml')

articlesDict = defaultdict(list)

for article in soup.findAll(name='article'):
   headline = article.h2.a.text
   summary = article.find(name='div', class_='entry-content').p.text
   videoId = article.find(name='iframe', class_='youtube-player')['src'].split('/')[4].split('?')[0]
   videoSource = f'https://youtube.com/watch/{videoId}'

   articlesDict['headline'].append(headline)
   articlesDict['summary'].append(summary)
   articlesDict['videoSource'].append(videoSource)

df = pd.DataFrame(data=articlesDict, columns=['headline', 'summary', 'videoSource'])
print(df)