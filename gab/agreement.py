import pandas as pd
import numpy as np
import json

df = pd.read_table('GabHateCorpus_annotations.tsv')
df = df[['ID','Annotator', 'Text', 'Hate','HD','CV','VO']]
# print(df.Annotator.value_counts())
# print(df.groupby('Annotator')['ID'].count())
# print(df.groupby(['Annotator','CV'])['ID'].count())
# print(df.groupby(['Annotator','CV'])[['Hate','HD','VO']].sum())

annotators = df['Annotator'].unique()
cats = ['Hate','HD','CV','VO']

# A function that looks at how an annotator scored a specific comment versus the average of everyone else on that comment
# average these difference values across all comments for each score category.

def agree(ann, cat, df):
    df_ann = df[df['Annotator'] == ann]

    df_other_mean = df[df['Annotator'] != ann].groupby('ID', as_index=False)[cat].mean().rename(columns={cat: 'mean_other'})

    df_merged = df_ann.merge(df_other_mean, on='ID', how='left')

    diffs = np.abs(df_merged[cat] - df_merged['mean_other'])

    return round(np.mean(diffs),4)


def scoring(anns, cats, df):
    """
    Loops through each annotator and scores each annotator against all others for each category.
    :param anns: list of annotators
    :param cats: list of categories
    :return: nested dict of scores for all annotators and categories
    """
    agree_scores = {str(ann): {cat: agree(ann, cat, df) for cat in cats} for ann in anns}

    return agree_scores

scores=scoring(annotators, cats, df)

# save scores to .txt document
with open("scores.json", "w") as outfile:
    json.dump(scores, outfile)