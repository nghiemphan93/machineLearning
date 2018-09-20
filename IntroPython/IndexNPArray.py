import numpy as np
import matplotlib.pyplot as plt
import seaborn
import urllib.request

#urllib.request.urlretrieve("ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt", "stations.txt")
data = open("stations.txt", "r").readlines()

print(data[:10])
