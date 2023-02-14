
'''
Description: Takes in a list of users and performs dimension reduction, clustering, and keyword extraction on the relevant tweet data.
Author: Tyler J Burns
Date: February 14, 2023
'''

from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
import nltk
import sklearn.cluster as cluster
import pandas as pd
import umap

users = pd.read_csv('src/users_to_display.csv')['user'].tolist()
print(users)

df = []
for i in users:
    curr = pd.read_feather('data/embedded_tweets/' + i + '_sentence_embeddings.feather')
    #curr = curr.head(5000)
    df.append(curr)

df = pd.concat(df).reset_index()

cols = [str(i) for i in range(768)] # Number of dimensions in the embedding

st = df[cols]
embedding = umap.UMAP(densmap=True, random_state=42).fit_transform(st)
embedding = pd.DataFrame(embedding, columns = ['umap1', 'umap2'])

# remove the sentence embedding bulk
df = df.drop(cols, axis = 1)

# Adds umap2 and umap2 as columns, and overwrites the orignal feather file
df = pd.concat([df.reset_index(drop=True), embedding], axis=1)

# TODO solve the hdbscan bug. 
# Relevant link: https://stackoverflow.com/questions/73830225/init-got-an-unexpected-keyword-argument-cachedir-when-importing-top2vec
clust_method = 'kmeans'

# Dev
# df = df.head(10000)

# We are using dbscan rather than hdbscan at the moment due to a bug from one of the libraries being the wrong version. 
if clust_method == "dbscan":
    
    '''
    Hdbscan is busted and has not been updated
    import hdbscan
    labels = hdbscan.HDBSCAN(
        min_samples=5,
        min_cluster_size=20,
    ).fit_predict(dimr[['umap1', 'umap2']])
    '''
    labels = cluster.DBSCAN(eps=3, min_samples=2).fit_predict(df[['umap1', 'umap2']])
else:
    labels = cluster.KMeans(n_clusters=50).fit_predict(df[['umap1', 'umap2']])
    
    # We want meanshift but it runs too slow for our larger datasets. 
    # TODO find a faster version of meanshift. Use cython or numba?
    # labels = cluster.MeanShift(min_bin_freq = 100).fit_predict(dimr[['umap1', 'umap2']])

df['cluster'] = labels
print(df.cluster[1:10])

# For initial use of the script
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

# Here is the keyword extraction. We're using KeyBERT simply because it's consistent with sBERT, which we use to do the embeddings
# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
kw_model = KeyBERT()

keywords_df = []
for i in np.unique(df['cluster']):
    curr = df[df['cluster'] == i] 
    text =  ' '.join(curr['Tweet']) 
    
    # Lemmatization
    text = nltk.word_tokenize(text)
    text = [wnl.lemmatize(i) for i in text]
    text = ' '.join(text)
    
    # Keyword extraction
    TR_keywords = kw_model.extract_keywords(text)
    keywords_df.append(TR_keywords[0:10])
    
keywords_df = pd.DataFrame(keywords_df)
keywords_df['cluster'] = np.unique(df['cluster'])
keywords_df.columns = ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'cluster']
print(keywords_df)

# Combine with original dataframe
df = df.merge(keywords_df, on = 'cluster', how = 'left') # This messes up the index

print(df.cluster[1:10])
print(df.keyword1[1:10])

df.to_csv('src/tmp.csv', index = False)