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
produkte       = "C:/Users/phan/OneDrive - adesso Group/bofrost/export-Produkte"
tour           = "C:/Users/phan/OneDrive - adesso Group/bofrost/export-Kunden_TourenDaten_50k_DE"
produkte       = pd.read_csv(produkte)
tour           = pd.read_csv(tour, parse_dates=["liefer_dtm"])
# endregion


# region Manipulate Data
tour                = tour.sort_values("liefer_dtm")
tour["dayOfWeek"]   = tour["liefer_dtm"].dt.dayofweek
tour["dayName"]     = tour["liefer_dtm"].dt.day_name('deu')
tour["monthOfYear"] = tour["liefer_dtm"].dt.month
tour["monthName"]   = tour["liefer_dtm"].dt.month_name('deu').str[:3]
# endregion



'''
tour[(tour['mitarbeiter_oid'] == 17700155) & (tour['liefer_dtm'] == '2019-03-05')]
tour[tour['liefer_dtm'] == '2019-03-05'].groupby('mitarbeiter_oid').value_counts()
tour = tour.dropna(subset='time_stamp')
'''


