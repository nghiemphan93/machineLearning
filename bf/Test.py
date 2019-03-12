# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
import calendar, locale
locale.setlocale(locale.LC_ALL, 'deu_deu')
print(type(list(calendar.day_name)))
# endregion

## Import Data
sales = "C:/Users/phan/OneDrive - adesso Group/bofrost/SalesRandomAnonym.csv"
tour = "C:/Users/phan/OneDrive - adesso Group/bofrost/TourenDaten_Anonimisiert.csv"
sales = pd.read_csv(sales, parse_dates=["DAT"])
tour = pd.read_csv(tour, parse_dates=["tw_dtm", "bearbeitungszeitpunkt"])

## Manipulate Data
def dayToText(dayIndex: int):
   dayNames = list(calendar.day_name)
   return dayNames[dayIndex]
def monthToText(monthIndex: int):
   monthNames = list(calendar.month_name)
   return monthNames[monthIndex]

sales["dayOfWeek"] = sales["DAT"].dt.dayofweek
sales["dayName"] = sales["dayOfWeek"].apply(dayToText)
sales["monthOfYear"] = sales["DAT"].dt.month
sales["monthName"] = sales["monthOfYear"].apply(monthToText)

'''
tour["hour"] = tour["bearbeitungszeitpunkt"].dt.hour
tour["month"] = tour["bearbeitungszeitpunkt"].dt.month
tour["dayName"] = tour["tw_dtm"].dt.weekday_name
tour["dayOfWeek"] = tour["bearbeitungszeitpunkt"].dt.dayofweek
'''
print(sales.sample(5))





## Visualize Data
# Anzahl der Bestellungen in der Woche
'''
plt.figure(figsize=(8, 5))
ax: plt.Axes = sns.countplot(data=sales.sort_values(by="dayOfWeek"), x="dayName")
ax.set(xlabel="Tag der Woche", ylabel="Anzahl", title="Anzahl der Bestellungen in der Woche")
plt.show()
'''


'''
plt.figure(figsize=(12, 10))
#sns.countplot(data=tour, x="ergebnis", hue="fixtermin")
ax: plt.Axes = sns.countplot(data=tour, x="fixtermin", hue="ergebnis")
plt.show()
'''

'''
sns.countplot(data=tour, x="fixtermin")
plt.show()
'''
'''
fig = plt.figure(figsize=(12, 5))
sns.countplot(x="EXT_mitarbeiter", data=tour)
plt.show()
'''
'''



plt.figure(figsize=(12, 5))
sns.countplot(x="hour", data=tour)
plt.show()
'''