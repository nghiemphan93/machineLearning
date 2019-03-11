# region Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
plt.style.use("fivethirtyeight")
pd.set_option('display.expand_frame_repr', False)
# endregion

sales = "C:/Users/phan/OneDrive - adesso Group/bofrost/SalesRandomAnonym.csv"
tour = "C:/Users/phan/OneDrive - adesso Group/bofrost/TourenDaten_Anonimisiert.csv"

sales = pd.read_csv(sales, parse_dates=["DAT"])
tour = pd.read_csv(tour, parse_dates=["tw_dtm", "bearbeitungszeitpunkt"])

print(sales)
print(tour)


plt.figure(figsize=(12, 10))
#sns.countplot(data=tour, x="ergebnis", hue="fixtermin")
sns.countplot(data=tour, x="fixtermin", hue="ergebnis")
plt.show()

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
tour["hour"] = tour["bearbeitungszeitpunkt"].dt.hour
print(tour)


plt.figure(figsize=(12, 5))
sns.countplot(x="hour", data=tour)
plt.show()
'''