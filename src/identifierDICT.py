import pandas as pd
import numpy as np

#Reads the Excel file
df = pd.read_excel('data/CollDataWAG.xlsx')

#Drops the rows where the in the subset specified information is missing
df = df.dropna(how='any', subset=['Identifier: (Resource Information)', 'Main Title: (Title)/Main Title: (Title Details)', 'Creator\'s Name'])

df.drop(['Accession No: (Accession Details)', 'Object Name: (Object Details)', 'Main Title: (Title)/Main Title: (Title Details)', 'Creator\'s Name',
    'Date of Birth',
    'Date of Death',
    'Date Created: (Creation Details)',
    'Earliest: (Creation Details)',
    'Latest: (Creation Details)',
    'Technique: (Technique and Material)',
    'Medium: (Technique and Material)',
    'Material: (Technique and Material)',
    'Description (Physical 1)',
    'Internal Record Number'], axis=1, inplace=True)

#Detects missing values in the dataframe
df.isnull().sum()



print(df.isnull().sum())

print(df.dropna())

#Prints the length of the index (rows)
print(len(df.index))


print(df.drop)

df.to_csv('WAG_identifier.csv', index = False)

