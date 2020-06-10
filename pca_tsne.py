import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Set plot params
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize

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
    # TODO Generate with multiple perplexity values and compare - https://distill.pub/2016/misread-tsne/
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
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=5000)
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
month = 'March'
month_df = df.loc[df['Month'] == month]
num_cats = len(df["Geo_Location"].unique())
conts = month_df['Continent'].unique()
num_conts = len(month_df["Continent"].unique())
num_months = len(df["Month"].unique())
color = sns.color_palette("colorblind", num_conts)
# plot
plt.figure(figsize=(12,9))
for i, cont in enumerate(conts):
    cm = color[i]
    tmp_df = month_df.loc[month_df['Continent'] == cont]
    if len(tmp_df['tsne-2d-one'])>2:
        print(cont)
        sns.kdeplot(data=tmp_df['tsne-2d-one'], 
                    data2=tmp_df['tsne-2d-two'],
                    shade=True,
                    shade_lowest=False,
                    alpha=0.4,
                    color=cm,
                    n_levels=5
        )
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #palette=sns.cubehelix_palette(num_cats),
    palette=sns.color_palette("colorblind", num_conts),
    #palette = sns.color_palette(flatui, num_months),
    style="Continent",
    hue="Continent",
    hue_order=conts,
    data=month_df,
    legend="full",
    alpha=0.8,
    linewidth=0.5,
    s=60
)



# Remove old legend
ax = plt.gca()
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend_ = None
ax.set_title(f'Covid Cluster - {month} 2020')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
plt.legend(handles, labels, ncol=2, loc="center left", bbox_to_anchor=(0.8, 0.95))
# plt.tight_layout()
plt.savefig(f'figures/{month}_t_sne.png')


# plot based on countries/continents - colors from https://material.io/design/color/the-color-system.html#tools-for-picking-colors
# unnecessary way of doing this, but it should work at least
continent_styles = {
    'Americas' : "o", # Cirlce
    'Europe' : "^",   # Triangle
    'Asia' :  "p", # Plus
    'Oceania' : "x", # Cross
    'Africa' : "D", # Diamond
}
country_colors = {
    'Americas' : ['#880E4F', '#AD1457', '#C2185B', '#D81B60', '#E91E63', '#EC407A', '#F06292'], # Pink
    'Europe' : ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#E8F5E9', '#00C853', '#00E676'], # Green
    'Asia' : ['#BF360C', '#D84315', '#E64A19', '#F4511E', '#FF5722', '#FF7043', '#FF8A65', '#FFAB91', '#FF6D00', '#FF9100', '#FFAB40', '#FFD180', '#E65100', '#EF6C00', '#F57C00', '#FB8C00', '#FF9800'], # Orange
    'Oceania' : ['#1A237E', '#283593'], # Blue
    'Africa' : ['#880E4F'] # Purple
}

country_map = {
    "Americas" : {
        'USA' : '#880E4F',
        'Puerto Rico' : '#AD1457',
        'Jamaica' : '#C2185B',
        'Uruguay' : '#D81B60',
        'Peru' : '#E91E63',
        'Brazil' : '#EC407A',
        'Colombia' : '#F06292',
    },
    "Europe" : {
        'France' : '#1B5E20',
        'Poland' : '#2E7D32',
        'Greece' : '#388E3C',
        'Czech Republic' : '#43A047',
        'Germany' : '#66BB6A',
        'Italy' : '#81C784',
        'Finland' : '#A5D6A7',
        'Netherlands' : '#C8E6C9',
        'Serbia' : '#E8F5E9',
        'Sweden' : '#00C853',
        'Spain' : '#00E676',
        'Turkey' : '#69F0AE',
    },
    "Asia" : {
        'India' : '#BF360C',
        'Japan' : '#D84315',
        'South Korea' : '#E64A19',
        'Taiwan' : '#F4511E',
        'China' : '#FF7043',
        'Sri Lanka' : '#FF8A65',
        'Hong Kong' : '#FFAB91',
        'Thailand' : '#FF6D00',
        'Viet Nam' : '#FF9100',
        'Israel' : '#FFAB40',
        'Malaysia' : '#FFD180',
        'Pakistan' : '#E65100',
        'Iran' : '#EF6C00',
        'Bangladesh' : '#F57C00',
        'Russia' : '#FB8C00',
        'Kazakhstan' : '#FF9800',
        'Nepal' : '#FFB74D'
    },
    "Oceania" : {
        'Australia' : '#1A237E',
        'Guam' : '#283593'
    },
    "Africa" : {
        'South Africa' : "#880E4F"
    },
}

# Run map back and pair each country with a color 
fig, ax = plt.subplots(figsize=(8,6))
last_continent = ""
t = 0
for idx, row in month_df.iterrows(): #! Note - monthdf
    continent = row['Continent']
    country = row['Country']
    if not continent == last_continent:
        plot_label = True
        i = 0 # Reset index in the subcountry color map
        last_continent = continent
    else:
        plot_label = False
        i+=1
    if country == 'Unknown':
        color = "#4E342E" # Brown
        style = '*'
    else: 
        color = country_map[continent][country]
        style = continent_styles[continent]
    # Plot
    if plot_label:
        ax.scatter(row['tsne-2d-one'], row['tsne-2d-two'], alpha=1, c=color, marker=style , label=f'{continent}')
    else:
        ax.scatter(row['tsne-2d-one'], row['tsne-2d-two'], marker=style, alpha=1, c=color)
    t += 1
ax.legend(loc='best')
ax.set_xlabel('tsne dimension one')
ax.set_ylabel('tsne dimension two')
ax.set_title(f'Clustering of Continent/Country - {month} 2020')
plt.tight_layout()
plt.savefig(f'figures/{month}_continent_country.pdf')
