'''
Description: Takes in a list of users and outputs a tmp.feather object that contains all of the user tweet sentence embeddings.
Author: Tyler J Burns
Date: November 2, 2022
'''

import pandas as pd

users = pd.read_csv('users_combo.csv')['user'].tolist()
print(users)

output = []
for i in users:
    curr = pd.read_feather('../transform_sentences/output/' + i + '_sentence_embeddings.feather')
    #curr = curr.head(5000)
    output.append(curr)

print(output)

output = pd.concat(output).reset_index()
output.to_feather('tmp.feather')