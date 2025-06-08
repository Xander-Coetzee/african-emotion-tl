from datasets import load_dataset
import pandas as pd

def load_and_preprocess_brighter(language):
    # this dataset below is the one with intensity levels for emotions -> 
    # we can decide later on which to use 
    # but we have less languages to choose from specifically only Arabic is available as an African language to train on
    # dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-intensities", language)

    # this data set uses standard binary classification for emotion categories - 1 present, 0 absent
    dataset = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", language)

    # we have 3 datasets available - train, test and dev
    # does it make sense to pre process them seperately and choose as needed?
    print(f" Dataset loaded with splits: {list(dataset.keys())}")

    def clean_split(split_df):
        df = split_df.to_pandas()
        # basic cleaning: drop rows where 'text' is missing or empty
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']
        return df

    cleaned_splits = {
        split_name: clean_split(dataset[split_name])
        for split_name in dataset
    }

    return cleaned_splits