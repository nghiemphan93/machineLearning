# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion


# region Import Data
sales    = "C:/Users/phan/OneDrive - adesso Group/bofrost/SalesRandomAnonym.csv"
tour     = "C:/Users/phan/OneDrive - adesso Group/bofrost/TourenDaten_Anonimisiert.csv"
sales    = pd.read_csv(sales, parse_dates=["DAT"])
tour     = pd.read_csv(tour, parse_dates=["tw_dtm", "bearbeitungszeitpunkt"])
# endregion


# region Manipulate Data
sales["dayOfWeek"]   = sales["DAT"].dt.dayofweek
sales["dayName"]     = sales["DAT"].dt.day_name('deu')
sales["monthOfYear"] = sales["DAT"].dt.month
sales["monthName"]   = sales["DAT"].dt.month_name('deu').str[:3]

# Convert "bearbeitungszeitpunkt"
def isBearbeitet(temp: pd.datetime):
   if pd.isnull(temp):
      return "nicht bearbeitet"
   else:
      return "bearbeitet"

tour["hour"]         = tour["bearbeitungszeitpunkt"].dt.hour
tour["monthOfYear"]  = tour["tw_dtm"].dt.month
tour["monthName"]    = tour["tw_dtm"].dt.month_name('deu').str[:3]
tour["dayName"]      = tour["tw_dtm"].dt.day_name('deu')
tour["dayOfWeek"]    = tour["tw_dtm"].dt.dayofweek
tour["bearbeitet"]   = tour["bearbeitungszeitpunkt"].apply(isBearbeitet)

tour = tour.sort_values(by="tw_dtm")
tourOneYear: pd.DataFrame = tour.loc[:318500, :]   # select one whole year

#print(sales.sample(10))
print(tourOneYear.sample(10))
print(tourOneYear["bearbeitet"].value_counts())
# endregion


# region Visualize Data
# Stundenverteilung der Ergebnisse
f: plt.Figure = plt.figure()
figName = "Stundenverteilung der Ergebnisse"
g = sns.FacetGrid(data=tourOneYear, row="ergebnis", aspect=4, sharey=False, hue="fixtermin")
g.map(sns.countplot, "hour")
g.set_xlabels("Uhrzeit")
g.set_ylabels("Anzahl")
g.add_legend()
plt.show()
g.savefig(figName)

# Wie viele verschiedene Monate
'''
f: plt.Figure = plt.figure(figsize=(10, 5))
figName = "Anzahl der verschiedenen Monate"
ax: plt.Axes = sns.countplot(data=sales, x="monthName")
ax.set(xlabel="Monat", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Bearbeitet wenn Nicht angetroffen
'''
f: plt.Figure = plt.figure(figsize=(10, 5))
figName = "Bearbeitet wenn Nicht angetroffen"
nichtAngetropffenDF = tourOneYear[tourOneYear["ergebnis"] == "Nicht angetroffen"]
ax: plt.Axes = sns.countplot(data=nichtAngetropffenDF, x="bearbeitet")
ax.set(xlabel="Bearbeitet", ylabel="Nicht angetroffen", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Verteilung der Bearbeitungszeit
'''
f: plt.Figure = plt.figure(figsize=(10, 5))
figName = "Verteilung der Bearbeitungszeit"
ax: plt.Axes = sns.countplot(data=tourOneYear, x="hour")
ax.set(xlabel="Stunde", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# FixTermin und Ergebnis
'''
f: plt.Figure = plt.figure(figsize=(10, 5))
figName = "Fixtermin und Ergebnis"
ax: plt.Axes = sns.countplot(data=tourOneYear, x="fixtermin", hue="ergebnis")
ax.set(xlabel="FixTermin", ylabel="Anzahl", title=figName)
ax.legend(loc="upper right")
plt.show()
f.savefig(fname=figName)
'''

# Anzahl der Bestellung im Monat
'''
f: plt.Figure = plt.figure(figsize=(12, 5))
figName = "Anzahl der Bestellungen im Monat"
ax: plt.Axes = sns.countplot(data=tourOneYear.sort_values(by="monthOfYear"), x="monthName")
ax.set(xlabel="Monat im Jahr", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Wie viele verschiedene Tage
'''
newDF = pd.DataFrame(sales["DAT"].value_counts())
newDF = newDF.reset_index()
newDF["dayName"] = newDF["index"].dt.day_name('deu')
temp = pd.DataFrame(newDF["dayName"].value_counts())
temp = temp.loc[["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]]
temp["dayName"] = temp["dayName"].astype(int)

f: plt.Figure = plt.figure(figsize=(8, 5))
figName = "Anzahl der verschiedenen Tage"
sns.barplot(data=temp, x=temp.index, y="dayName")
ax.set(xlabel="Tag der Woche", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Anzahl der Bestellungen in der Woche
'''
f: plt.Figure = plt.figure(figsize=(8, 5))
figName = "Anzahl der Bestellungen der Woche"
ax: plt.Axes = sns.countplot(data=sales.sort_values(by="dayOfWeek"), x="dayName")
ax.set(xlabel="Tag der Woche", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Anzahl der Bestellungen im Jahr
'''
f: plt.Figure = plt.figure(figsize=(8, 5))
figName = "Anzahl der Bestellungen im Jahr"
ax: plt.Axes = sns.countplot(data=sales.sort_values(by="monthOfYear"), x="monthName")
ax.set(xlabel="Monat des Jahres", ylabel="Anzahl", title=figName)
plt.show()
f.savefig(fname=figName)
'''

# Umsatz in der Woche
'''
byDayName: pd.DataFrame = sales.groupby("dayName").sum()
byDayName = byDayName.loc[["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"], :]
byDayName = byDayName.reset_index()
print(byDayName.head())

f: plt.Figure = plt.figure(figsize=(8, 5))
figName = "Umsatz in der Woche"
ax: plt.Axes = sns.barplot(data=byDayName, x="dayName", y="DELIVERY_PRICE_EUR")
ax.set(xlabel="Tage der Woche", ylabel="Umsatz", title=figName)
plt.show()
f.savefig(fname=figName)
'''
# endregion

