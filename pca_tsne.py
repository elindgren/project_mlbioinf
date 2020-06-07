import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

map = {'USA': 'Americas', 'Australia': 'Oceania', 'France': 'Europe', 'Poland': 'Europe', 'India': 'Asia', 'Greece': 'Europe', 'Japan': 'Asia',
        'Czech Republic': 'Europe', 'South Korea': 'Asia', 'Taiwan': 'Asia', 'Puerto Rico': 'Americas', 'China': 'Asia',
        'Sri Lanka': 'Asia', 'Germany': 'Europe', 'Italy': 'Europe', 'Hong Kong': 'Asia', 'Finland': 'Europe',
        'Thailand': 'Asia', 'Jamaica': 'Americas', 'Spain': 'Europe', 'Viet Nam': 'Asia', 'Netherlands': 'Europe',
        'Israel': 'Asia', 'Malaysia': 'Asia', 'Guam': 'Oceania', 'Pakistan': 'Asia', 'Unknown': 'Unknown', 'Iran': 'Asia',
        'Uruguay': 'Americas', 'Bangladesh': 'Asia', 'Russia': 'Asia', 'South Africa': 'Africa', 'Colombia': 'Americas',
        'Serbia': 'Europe', 'Kazakhstan': 'Asia', 'Peru': 'Americas', 'Brazil': 'Americas', 'Nepal': 'Asia', 'Sweden': 'Europe',
        'Turkey': 'Europe'}

# load data
df = pd.read_pickle("files/merged_100.pickle")

# add continents
continent = []
for idx, row in df.iterrows():
    if pd.notnull(df.at[idx, "Geo_Location"]):
        location = df.at[idx,'Geo_Location']
    else:
        location = "Unknown"
    if ":" in location:
        location, _ = location.split(":")
    #df.at[idx,'Geo_Location'] = map[location]
    continent.append(map[location])

df["Continent"] = continent
df = df.sort_values(by=["Continent"], ascending=False)

print(df)

data = np.stack(df["z"].values)

# do pca to reduce down to 50 dimensions
pca = PCA(n_components=50)
pca_result = pca.fit_transform(data)
print('Cumulative explained variation for principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

# keep these to visualize the PCA results
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 

# get number of unique categories to choose graph colors
num_cats = len(df["Geo_Location"].unique())
# make plot
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    palette=sns.color_palette("colorblind", num_cats),
    style="Continent",
    hue="Geo_Location",
    data=df,
    legend="full",
    alpha=0.7
)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=60, n_iter=1500)
tsne_results = tsne.fit_transform(data) # use original data
#tsne_results = tsne.fit_transform(pca_result) # use pca data
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

# remove countries with less than n data points
#df = df.groupby('Geo_Location').filter(lambda x : len(x)>5)

# get number of unique categories to choose graph colors
num_cats = len(df["Geo_Location"].unique())
# plot
plt.figure(figsize=(20,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #palette=sns.cubehelix_palette(num_cats),
    palette=sns.color_palette("colorblind", num_cats),
    style="Continent",
    hue="Geo_Location",
    data=df,
    legend="full",
    alpha=0.5,
    linewidth=0.5
)

# Remove old legend
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend_ = None
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
plt.legend(handles, labels, ncol=2, loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()
