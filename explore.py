# Data exploration
import pandas as pd
import numpy as np
import json


# df = pd.read_table('gab/GabHateCorpus_annotations.tsv')
#
# print(df.Annotator.value_counts())
# print(df.groupby('Annotator')['ID'].count())
# print(df.groupby(['Annotator','CV'])['ID'].count())
# print(df.groupby(['Annotator','CV'])[['Hate','HD','VO']].sum())

with open('gab/scores.json') as f:
    scores = json.load(f)

df = pd.DataFrame.from_dict(scores, orient='index')

df.to_csv('gab/scores.csv')
