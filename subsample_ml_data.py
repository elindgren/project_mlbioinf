import os
from tqdm import tqdm

covid_samples = {
    'NC_045512': 'China', 
    'MT407649' : 'China',
    'MT407654' : 'China',
    'MT407658' : 'China',
    'MT135043' : 'China',
    'MT534303' : 'USA',
    'MT534328' : 'USA',
    'MT535498' : 'USA',
    'MT536185' : 'USA',
    'MT528605' : 'USA',
    'MT525950' : 'Italy',
    'MT527178' : 'Italy',
    'MT527184' : 'Italy',
    'MT528235' : 'Italy',
    'MT528237' : 'Italy',
    'MT270103' : 'Germany',
    'MT270104' : 'Germany',
    'MT270105' : 'Germany',
    'MT270108' : 'Germany',
    'MT270109' : 'Germany',
    'MT483554' : 'India',
    'MT483556' : 'India',
    'MT483558' : 'India',
    'MT483559' : 'India',
    'MT483560' : 'India',
    'LC547518' : 'Japan',
    'LC547520' : 'Japan',
    'LC547521' : 'Japan',
    'LC547522' : 'Japan',
    'LC547523' : 'Japan',
}

mers_samples = {
    'MN481964' : 'Saudi Arabia',
    'MN481965' : 'Saudi Arabia',
    'MN481966' : 'Saudi Arabia',
    'MN481977' : 'Saudi Arabia',
    'MN481985' : 'Saudi Arabia',
    'MK039552' : 'Jordan',
    'MK039553' : 'Jordan',
    'MF000458' : 'Jordan',
    'MF741826' : 'Jordan',
    'MF741836' : 'Jordan',
    'KT326819' : 'South Korea',
    'KX034094' : 'South Korea',
    'KX034096' : 'South Korea',
    'KX034098' : 'South Korea',
    'KT868866' : 'South Korea',
    'KJ361499' : 'France',
    'KJ361500' : 'France',
    'KJ361501' : 'France',
    'KJ361502' : 'France',
    'KF745068' : 'France',
    'MH822886' : 'United Kingdom',
    'KM210277' : 'United Kingdom',
    'KM210278' : 'United Kingdom',
    'KM015348' : 'United Kingdom',
    'KC667074' : 'United Kingdom',
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
            
                

                
                

