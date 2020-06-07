import os
import pickle
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

month_map = {
    '01' : 'January',
    '02' : 'February',
    '03' : 'March',
    '04' : 'April',
    '05' : 'May',
    '06' : 'June',
    '07' : 'July',
    '08' : 'August',
    '09' : 'September',
    '10' : 'October',
    '11' : 'November',
    '12' : 'December',
}

# load data
sf = "merged_50"
df = pd.read_pickle(f'files/{sf}.pickle')


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

# add countries
countries = []
for idx, row in df.iterrows():
    if pd.notnull(df.at[idx, "Geo_Location"]):
        location = df.at[idx,'Geo_Location']
    else:
        location = "Unknown"
    if ":" in location:
        location, _ = location.split(":")
    countries.append(location)
df["Country"] = countries

# add months
months = []
for idx, row in df.iterrows():
    if pd.notnull(df.at[idx, "Collection_Date"]):
        date = df.at[idx, "Collection_Date"]
    else:
        date = "Unknown"
    if "-" in date:
        month = date.split('-')[1]
        months.append(month_map[month])
    else: 
        months.append(date)
    
df["Month"] = months
# df = df.sort_values(by=["Month"], ascending=False)
print(df)
data = np.stack(df["z"].values)

# do pca to reduce down to 50 dimensions
pca = PCA(n_components=25)
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

if not os.path.isfile(f't_SNE-{sf}.pickle'):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=600)
    tsne_results = tsne.fit_transform(data) # use original data
    #tsne_results = tsne.fit_transform(pca_result) # use pca data
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    with open(f't_SNE-{sf}.pickle','wb') as f:
        pickle.dump(tsne_results,f)
else: 
    with open(f't_SNE-{sf}.pickle','rb') as f:
        tsne_results = pickle.load(f)


df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

# remove countries with less than n data points
#df = df.groupby('Geo_Location').filter(lambda x : len(x)>5)

# get number of unique categories to choose graph colors
num_cats = len(df["Geo_Location"].unique())
num_conts = len(df["Continent"].unique())
num_months = len(df["Month"].unique())
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# plot
plt.figure(figsize=(20,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #palette=sns.cubehelix_palette(num_cats),
    # palette=sns.color_palette("colorblind", num_months),
    palette = sns.color_palette(flatui, num_months),
    style="Continent",
    hue="Month",
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
plt.savefig('t_sne.png')


# plot based on countries/continents - colors from https://material.io/design/color/the-color-system.html#tools-for-picking-colors
# unnecessary way of doing this, but it should work at least
continent_colors = {
    'Americas' : "#880E4F", # Pink
    'Europe' : "#1B5E20",   # Green
    'Asia' :  "#BF360C", # Orange
    'Oceania' : "#1A237E", # Blue
    'Africa' : "#880E4F", # Purple
}
country_colors = {
    'Americas' : ['#880E4F', '#AD1457', '#C2185B', '#D81B60', '#E91E63', '#EC407A', '#F06292'], 
    'Europe' : ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#E8F5E9', '#00C853', '#00E676'], 
    'Asia' : ['#BF360C', '#D84315', '#E64A19', '#F4511E', '#FF5722', '#FF7043', '#FF8A65', '#FFAB91', '#FF6D00', '#FF9100', '#FFAB40', '#FFD180', '#E65100', '#EF6C00', '#F57C00', '#FB8C00', '#FF9800'],
    'Oceania' : ['#1A237E', '#283593'],
    'Africa' : ['#880E4F']
}

# Run map back and pair each country with a color 
fig, ax = plt.subplots(figsize=(8,6))
country_map = {}
last_continent = ""
for idx, row in df.iterrows():
    continent = row['Continent']
    country = row['Country']
    if not continent == last_continent:
        i = 0 # Reset index in the subcountry color map
    else:
        i+=1
    if country == 'Unknown':
        color = "#4E342E" # Brown
    else: 
        color = country_colors[continent][i]
    country_map[country] = color
    # Plot
    ax.scatter(row['tsne-2d-one'], row['tsne-2d-two'], alpha=0.5, c=color, label=f'{continent}/{country}')
ax.legend(loc='best')
ax.set_xlabel('tsne dimension one')
ax.set_xlabel('tsne dimension two')
plt.show()
