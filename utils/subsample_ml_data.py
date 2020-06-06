# Internal imports
import os
import random
import pprint

# External imports
import numpy as np
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

covid_all_patients = {}
covid_samples = {}

mers_all_patiens = {}
mers_samples = {}

random.seed(1)
Ncovid = 300 # Will probably be a few more than this - since the program loops through all categories before checking this condition
Nmers = 300

print('Covid')
# Covid
with open('ml_data/covid19.csv', 'r') as d:
    for i, line in enumerate(d):
        if i > 0:
            sl = line.rstrip().split(',')
            ID = sl[0]
            country = sl[4]
            if ':' in country:
                country = country.split(':')[0]
            if '"' in country:
                country = country.split('"')[1]
            if country == '':
                country = 'N/A'
            if not country in covid_all_patients.keys():
                covid_all_patients[country] = [ID]
            else:    
                covid_all_patients[country].append(ID)
# Verify results and randomize order of lists
seqs = 0
for country, ids in covid_all_patients.items():
    seqs += len(ids)
    # Shuffle
    random.shuffle(ids)
    covid_all_patients[country] = ids
assert seqs == 4596
# Create list of patients - try to get the same number from each country
full = False
i = 0
Nadded = 0
while Nadded <= Ncovid:
    for country, ids in covid_all_patients.items():
        if not i > len(ids)-1:
            covid_samples[ids[i]] = country
            Nadded += 1
    i += 1
# Display sample information
stats = {}
s = 0
for ID, country in covid_samples.items():
    if not country in stats.keys():
        stats[country] = 1
    else:    
        stats[country] += 1
print(f'Adding {Nadded} sequences to fasta file {s}')
pp.pprint(stats)
with open('ml_data/covid19.fasta', 'r') as d:
    with open('covid_subsampled.fasta', 'w') as f:  
        for line in tqdm(d):
            if '>' in line:
                covid_writing = False
                for tag, country in covid_samples.items():
                    if tag in line:   
                        covid_writing = True
                        f.write(line)  
                    else:
                        covid_writing = covid_writing or False
            elif covid_writing:
                f.write(line)


print('********')

print('MERS')
# MERS
with open('ml_data/mers.csv', 'r') as d:
    for i, line in enumerate(d):
        if i > 0:
            sl = line.rstrip().split(',')
            ID = sl[0]
            country = sl[4]
            if ':' in country:
                country = country.split(':')[0]
            if '"' in country:
                country = country.split('"')[1]
            if country == '':
                country = 'N/A'
            if not country in mers_all_patiens.keys():
                mers_all_patiens[country] = [ID]
            else:    
                mers_all_patiens[country].append(ID)
# Verify results and randomize order of lists
seqs = 0
for country, ids in mers_all_patiens.items():
    seqs += len(ids)
    # Shuffle
    random.shuffle(ids)
    mers_all_patiens[country] = ids
assert seqs == 318
# Create list of patients - try to get the same number from each country
full = False
i = 0
Nadded = 0
while Nadded <= Nmers:
    for country, ids in mers_all_patiens.items():
        if not i > len(ids)-1:
            mers_samples[ids[i]] = country
            Nadded += 1
    i += 1
# Display sample information
stats = {}
for ID, country in mers_samples.items():
    if not country in stats.keys():
        stats[country] = 1
    else:    
        stats[country] += 1
print(f'Adding {Nadded} sequences to fasta file')
pp.pprint(stats)

with open('ml_data/mers.fasta', 'r') as d:
    with open('mers_subsampled.fasta', 'w') as f:  
        for line in tqdm(d):
            if '>' in line:
                mers_writing = False
                for tag, country in mers_samples.items():
                    if tag in line:   
                        mers_writing = True
                        f.write(line)  
                    else:
                        mers_writing = mers_writing or False
            elif mers_writing:
                f.write(line)
            
                

                
                

