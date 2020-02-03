# region Import
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import datetime as dt

sns.set()
pd.set_option('display.expand_frame_repr', False)
# endregion

filePickle = 'C:/Users/phan/OneDrive - adesso Group/bofrost-data/F_umsatz_kopf_49-resized.pickle'
with open(file=filePickle, mode='rb') as f:
   order: pd.DataFrame = pickle.load(f)

filePickle = 'C:/Users/phan/OneDrive - adesso Group/bofrost-data/f_umsatz_pos_50k_49-resized.pickle'
with open(file=filePickle, mode='rb') as f:
   detail: pd.DataFrame = pickle.load(f)

filePickle = 'C:/Users/phan/OneDrive - adesso Group/bofrost-data/KundeSelect-resized.pickle'
with open(file=filePickle, mode='rb') as f:
   kunde: pd.DataFrame = pickle.load(f)

filePickle = 'C:/Users/phan/OneDrive - adesso Group/bofrost-data/kunden_oid-liefer_oid'
with open(file=filePickle, mode='rb') as f:
   kundenLiefer: pd.DataFrame = pickle.load(f)

with open(file='C:/Users/phan/OneDrive - adesso Group/bofrost-data/kunden_oid-liefer_oid', mode='wb') as f:
   pickle.dump(f)

order.pivot_table