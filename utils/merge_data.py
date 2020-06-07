import pandas as pd

df = pd.read_pickle("../files/latents_100.pickle")
df_meta = pd.read_csv("../files/covid_meta.csv")

# Fix formatting of accession
for idx, row in df.iterrows():
    name = df.at[idx, "Name"]
    if "." in name:
        name, _ = name.split(".")
    df.at[idx,"Name"] = name

map = {'USA': 'Americas', 'Australia': 'Oceania', 'France': 'Europe', 'Poland': 'Europe', 'India': 'Asia', 'Greece': 'Europe', 'Japan': 'Asia',
        'Czech Republic': 'Europe', 'South Korea': 'Asia', 'Taiwan': 'Asia', 'Puerto Rico': 'Americas', 'China': 'Asia',
        'Sri Lanka': 'Asia', 'Germany': 'Europe', 'Italy': 'Europe', 'Hong Kong': 'Asia', 'Finland': 'Europe',
        'Thailand': 'Asia', 'Jamaica': 'Americas', 'Spain': 'Europe', 'Viet Nam': 'Asia', 'Netherlands': 'Europe',
        'Israel': 'Asia', 'Malaysia': 'Asia', 'Guam': 'Oceania', 'Pakistan': 'Asia', 'Unknown': 'Unknown', 'Iran': 'Asia',
        'Uruguay': 'Americas', 'Bangladesh': 'Asia', 'Russia': 'Asia', 'South Africa': 'Africa', 'Colombia': 'Americas',
        'Serbia': 'Europe', 'Kazakhstan': 'Asia', 'Peru': 'Americas', 'Brazil': 'Americas', 'Nepal': 'Asia', 'Sweden': 'Europe',
        'Turkey': 'Europe'}

# Fix formatting of country names
for idx, row in df_meta.iterrows():
    if pd.notnull(df_meta.at[idx, "Geo_Location"]):
        location = df_meta.at[idx,'Geo_Location']
    else:
        location = "Unknown"
    if ":" in location:
        location, _ = location.split(":")
    #df_meta.at[idx,'Geo_Location'] = map[location]
    df_meta.at[idx,'Geo_Location'] = location

df = df.merge(df_meta, left_on="Name", right_on="Accession")

print(df)

df.to_pickle("../files/merged_100.pickle")
