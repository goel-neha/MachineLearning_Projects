import pyprind
import pandas as pd
import os
import numpy as np
import csv
import re


df1 = pd.DataFrame()
df2 = pd.DataFrame()
df = pd.DataFrame()
# combine the both text files into 1 csv files giving them labels.
# pbar = pyprind.ProgBar(50000)
#Negative reviews added in csv
with open('polarity.neg.txt', 'r', encoding='utf-8') as infile:
    txt = infile.read().splitlines()
for line in txt:
    df1 = df1.append([[line, 0]], ignore_index=True)
# print(df1)
#Positive reviews added in csv
with open('polarity.pos.txt', 'r', encoding='utf-8') as infile2:
    txt2 = infile2.read().splitlines()
for line in txt2:
    df2 = df2.append([[line, 1]], ignore_index=True)

#Concatenation
df = pd.concat([df1, df2], axis=0)
# print(df)
df.columns = ['review', 'sentiment']

#shuffling the dataframe
from sklearn.utils import shuffle
df = shuffle(df)
#Movie.data csv file created
df.to_csv('./movie_data.csv', index=False)

# Splitting the dataset into the Training set , Dev set and Test set
msk = np.random.rand(len(df)) <= 0.7
dsk = np.random.rand(len(~msk)) <= 0.5

train = df[msk]
dev = df[dsk]
test = df[~dsk]

train.to_csv('./train.csv', index=False)
test.to_csv('./test.csv', index=False)
dev.to_csv('./dev.csv', index=False)
