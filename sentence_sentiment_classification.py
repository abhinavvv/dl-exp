import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# path = "dataset/sentiment labelled sentences"

import os
print(os.path.exists("datasets/sentiment_labelled_sentences"))

filepath_dict = {'yelp': "datasets/sentiment_labelled_sentences/yelp_labelled.txt",
                 'amazon': "datasets/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                 'imdb': "datasets/sentiment_labelled_sentences/imdb_labelled.txt"
                 }
df_list = []
for source, filepath in filepath_dict.items():
   df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
   # Add another column filled with the source name
   df['source'] = source
   df_list.append(df)
df = pd.concat(df_list)

df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train,sentences_test,y_train,y_test = train_test_split(
                                                sentences, y,
                                                test_size=0.25,
                                                random_state=1000)

print(len(sentences),df_yelp['sentence'].describe())