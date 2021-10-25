import pandas as pd
import numpy as np
import csv, json

"""
Reads the Excel file
"""
df = pd.read_excel('../data/original-data/MAG/CollDataMAG.xlsx', index_col=0)

"""
Detects missing values in the dataframe
"""
df.isnull().sum()

"""
Drops the rows where the in the subset specified information is missing
"""
df = df.dropna(how='any', subset=['Identifier', 'Main Title: (Title)/Main Title: (Title Details)', 'Creator\'s Name'])


print(df.isnull().sum())

print(df.dropna())

print(len(df.index))

df.to_csv('MAG.csv', index=False)

csvFile = open('MAG.csv', 'r')
jsonFile = open('MAGtest.json', 'w')



reader = csv.DictReader(csvFile)
jsonFile.write('[')
flag = False
for row in reader:
    if flag:
        jsonFile.write(',\n')
    json.dump(row, jsonFile)
    flag = True
jsonFile.write(']')

"""
Compare image data to see if it matches the meta-data
"""
f1 = pd.read_csv('MAG.csv')
f2 = pd.read_csv('../data/original-data/MAG/MAGimageID.csv')

f3 = (f1[~f1.Identifier.isin(f2.Identifier)])

f3.to_csv('MAGmatch.csv')