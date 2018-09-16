import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates


def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter

def graphData(stock):
    stockPriceUrl = "https://pythonprogramming.net/yahoo_finance_replacement"
    sourceCode = urllib.request.urlopen(stockPriceUrl).read().decode()


    stockData = []
    splitSource = sourceCode.split("\n")

    for line in splitSource:
        splitLine = line.split(",")
        if (len(splitLine) == 7):
            stockData.append(line)


    stockData.pop(0)
    print(stockData)

    '''
    date, openP, highP, lowP, closeP, adjustCloseP, volume = [],[],[],[],[],[],[]
    for i in range(len(stockData)):
        date, openP, highP, lowP, closeP, adjustCloseP, volume = np.loadtxt(stockData[i], delimiter=",", unpack=True)
    '''


    date, openP, highP, lowP, closeP, adjustCloseP, volume = np.loadtxt(stockData,
                                                                        delimiter=",",
                                                                        unpack=True,
                                                                        converters={0: bytespdate2num('%Y-%m-%d')})
    print(date)
    plt.plot_date(date, closeP, "-", label="Price")


    plt.xlabel("Plot Number")
    plt.ylabel("Important var")
    plt.title("Interesting Graph\nCheck it out")
    plt.legend()
    plt.show()




graphData("TSLA")