import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle('../../data/interim/02_outliers_removed_chauvenet.pkl')
predictor_coulumns = (df.columns[:6])

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
for col in predictor_coulumns:
    df[col] = df[col].interpolate()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
for s in df['set'].unique():
    start = df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]
    duration = stop - start
    df.loc[df['set']==s,"duration"] = duration.seconds
    
duration_df = df.groupby(['category'])["duration"].mean()

duration_df.iloc[0]/5
    
# --------------------------------------------------------------
# Butterworth lowpass filter
# -------------------------------------------------------------
df_lowpass = df.copy()
low_filter = LowPassFilter()
fs = 1000 / 200
fc = 1.3

df_lowpass= low_filter.low_pass_filter(df_lowpass , "acc_y" , fs , fc )
df_lowpass.info()

subset = df_lowpass[df_lowpass['set']==45]

fig,ax=plt.subplots(nrows=2,sharex=True,figsize=[20,10])
ax[0].plot(subset["acc_y"].reset_index(drop=True),label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True),label="lowpass_data")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)
ax[1].legend(loc="upper center" ,bbox_to_anchor=(0.5,1.15),fancybox=True,shadow=True)

for col in predictor_coulumns:
    df_lowpass = low_filter.low_pass_filter(df_lowpass, col, fs, fc)
    df_lowpass[col]= df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
pca= PrincipalComponentAnalysis()

pc_values = pca.determine_pc_explained_variance(df_pca, predictor_coulumns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(pc_values)+1),pc_values)
plt.xlabel("Principal component")
plt.ylabel("Explained variance")
plt.show()

df_pca = pca.apply_pca(df_pca, predictor_coulumns, 3)

subset = df_pca[df_pca['set']==35]
plt.plot(subset[["pca_1","pca_2","pca_3"]])
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)



# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_coulumns = list(predictor_coulumns) + ["acc_r","gyr_r"]

ws = int(1000/200)

for col in predictor_coulumns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []

for s in df_temporal['set'].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_coulumns:
        subset = NumAbs.abstract_numerical(subset,[col],ws,"mean")
        subset = NumAbs.abstract_numerical(subset,[col],ws,"std")
    df_temporal_list.append(subset)    
    
df_temporal= pd.concat(df_temporal_list)

df_temporal.info()


    
        


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy()
FreqAbs = FourierTransformation()

fs =int(1000/200)
ws =int(2800/200)

df_freq_list = []

for s in df_freq['set'].unique():
    print(f"Applying fourier transform to set ({s})")
    subset = df_freq[df_freq["set"]==s].reset_index().copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_coulumns, ws, fs)
    df_freq_list.append(subset)
    
df_freq= pd.concat(df_freq_list).set_index('epoch (ms)',drop=True)

df_freq.columns


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq=df_freq.dropna()

df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_colums = ["acc_x","acc_y","acc_z"]

k_values = range(2,10)

inertias = []

for k in k_values:
    subset = df_cluster[cluster_colums]
    kmeans =KMeans(n_clusters=k,n_init=20,random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("Number of clusters(k)")
plt.ylabel("Inertia")
plt.show()
    
kmeans =KMeans(n_clusters=k,n_init=20,random_state=0)
subset = df_cluster[cluster_colums]
df_cluster["Cluster"] = kmeans.fit_predict(subset)

plt.figure(figsize=(15,15))
ax = plt.add_subplot(projection='3d')
for c in df_cluster["Cluster"].unique():
    subset = df_cluster[df_cluster["Cluster"]==c]
    ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=c)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.legend()
plt.show()

plt.figure(figsize=(15,15))
ax = plt.add_subplot(projection='3d')
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["Cluster"]==c]
    ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=l)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_feature.pkl")