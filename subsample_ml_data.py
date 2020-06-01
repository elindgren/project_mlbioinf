import os
from tqdm import tqdm

covid_samples = {
    'NC_045512': 'China', 
    'MT407649' : 'China',
    'MT407650' : 'China',
    'MT534328' : 'USA',
    'MT534332' : 'USA',
    'MT358640' : 'Germany',
}

mers_samples = {
    'MN481964' : 'Saudi Arabia',
    'MF000457' : 'Jordan'
}
print('Covid')
# Covid
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
            
                

                
                

