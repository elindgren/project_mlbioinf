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
plt.rc('axes', labelsize=22)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize

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
df = df.sort_values(by=["Continent"], ascending=True)

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

perp = 20
if not os.path.isfile(f't_SNE-{sf}.pickle'):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=5000)
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

#* pick out accession numbers of interest
accesions = [
    'MT270104', 'MT270105', 'MT259229', 'MT259231', 'MT253700',
    'MT253699', 'MT449674', 'MT121215', 'MT407658', 'MT079844', 
    'MT079852', 'MN938384', 'MT528622', 'MT252799', 'MT326050', 
    'MT326108', 'MT292579', 'MT198651', 'MT292577', 'MT233521', 
    'MT292574', 'MT233523', 'MT256917', 'MT281577', 'MT077125',
    'MT292572', 'MT066156', 'MT019531', 'MT358641', 'MT358638', 
    'MT291828'
]
# df = df[df['Accession'].isin(accesions)]

#* get number of unique categories to choose graph colors
month = 'March'
month_df = df.loc[df['Month'] == month]

month_df = df

num_cats = len(df["Geo_Location"].unique())
conts = month_df['Continent'].unique()
num_conts = len(month_df["Continent"].unique())

color = sns.color_palette("colorblind", num_conts)

#* Create color map
color_map = {}
for i, cont in enumerate(conts):
    color_map[cont] = color[i]
continent_styles = {
    'Americas' : "x", # Cirlce
    'Europe' : "P",   # Triangle
    'Asia' :  "s", # Plus
    'Oceania' : "D", # Cross
    'Africa' : "o", # Diamond
    'Unknown' : "^"
}

#* plot
fig, ax = plt.subplots(figsize=(15,12))
grid = np.array(
    [
        np.linspace(month_df['tsne-2d-one'].min(), month_df['tsne-2d-one'].max(),20),
        np.linspace(month_df['tsne-2d-two'].min(), month_df['tsne-2d-two'].max(),20),
    ]
)
print(grid.shape)
dx = grid[0,1] - grid[0,0]
dy = grid[1,1] - grid[1,0]

np.random.seed(1)
#* Iterate over grid points, and calculate dominant 
added_labels = []
for i in range(grid.shape[1]-1):
    for j in range(grid.shape[1]-1):
        # Count the number of samples from each continent in each grid
        x = grid[0,i]
        y = grid[1,j]
        sub_df = month_df.loc[
            (month_df['tsne-2d-one'] >= x) & (month_df['tsne-2d-one'] <= x+dx) &
            (month_df['tsne-2d-two'] >= y) & (month_df['tsne-2d-two'] <= y+dy)
            ]
        if len(sub_df) > 0:
            unique, counts = np.unique(sub_df['Continent'], return_counts=True)
            dcont = ''
            size = counts[0]

            if size > 20000:
                # Plot dominant continent as blob if there are more than 20 of them in the block
                dcont = unique[0]
                ax.scatter(
                    x=x+dx/2, 
                    y=y+dy/2, 
                    color=color_map[dcont],
                    alpha=0.4,
                    marker="o",
                    s=size*20
                )
            # Scatter other points
            for idx, c_df in sub_df.loc[sub_df['Continent'] != dcont].iterrows():
                c = c_df['Continent']
                if not c in added_labels:
                    l = c
                    added_labels.append(c)
                else: 
                    l = ''
                ax.scatter(
                    x=c_df['tsne-2d-one'],
                    y=c_df['tsne-2d-two'],
                    color=color_map[c],
                    alpha=0.8,
                    marker=continent_styles[c],
                    edgecolor='w',
                    s=50,
                    label=l
                )
                
                # mx = np.random.uniform(1,5)
                # my = np.random.uniform(1,5)
                # mx = 1
                # my = 1
                # ax.text(
                #     x=c_df['tsne-2d-one']+mx,
                #     y=c_df['tsne-2d-two']+my,
                #     s=c_df['Accession'],
                #     fontsize=14,
                #     color='gray'
                # )


# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     #palette=sns.cubehelix_palette(num_cats),
#     palette=sns.color_palette("colorblind", num_conts),
#     #palette = sns.color_palette(flatui, num_months),
#     style="Continent",
#     hue="Continent",
#     hue_order=conts,
#     data=month_df,
#     legend="full",
#     alpha=0.8,
#     linewidth=0.5,
#     s=60
# )



# Remove old legend
# ax = plt.gca()
ax.grid()
handles, labels = ax.get_legend_handles_labels()
ax.legend_ = None
# ax.set_title(f'Covid Cluster - Full dataset 2020, perplexity={perp}')
ax.set_xlabel('t-SNE dimension one')
ax.set_ylabel('t-SNE dimension two')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
plt.legend(handles, labels, ncol=2, loc="center left", bbox_to_anchor=(0.8, 0.9))
# plt.tight_layout()
plt.savefig(f'figures/t_sne_perplexity{perp}.png')


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
# fig, ax = plt.subplots(figsize=(8,6))
# last_continent = ""
# t = 0
# for idx, row in month_df.iterrows(): #! Note - monthdf
#     continent = row['Continent']
#     country = row['Country']
#     if not continent == last_continent:
#         plot_label = True
#         i = 0 # Reset index in the subcountry color map
#         last_continent = continent
#     else:
#         plot_label = False
#         i+=1
#     if country == 'Unknown':
#         color = "#4E342E" # Brown
#         style = '*'
#     else: 
#         color = country_map[continent][country]
#         style = continent_styles[continent]
#     # Plot
#     if plot_label:
#         ax.scatter(row['tsne-2d-one'], row['tsne-2d-two'], alpha=1, c=color, marker=style , label=f'{continent}')
#     else:
#         ax.scatter(row['tsne-2d-one'], row['tsne-2d-two'], marker=style, alpha=1, c=color)
#     t += 1
# ax.legend(loc='best')
# ax.set_xlabel('tsne dimension one')
# ax.set_ylabel('tsne dimension two')
# ax.set_title(f'Clustering of Continent/Country - {month} 2020')
# plt.tight_layout()
# plt.savefig(f'figures/{month}_continent_country.pdf')
