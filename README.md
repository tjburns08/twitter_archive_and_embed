# Twitter archive and embed

# Description
A project that allows the user to pull the entire tweet history of any list of users, embed the tweets in a BERT language model, embed that into a UMAP, cluster the data, annotate the clusters based on keyword extraction, and visualize all of that in an interactive interface.

# How to use
1. Clone this repo and install the required python packages in requirements.txt. 
2. Add a data/ folder in the main directory. In that, add two folders: embedded_tweets/ and scraped_tweets/
3. Populate users.csv with specific twitter users you want to scrape and embed.
4. Run update.sh.
5. Populate users_to_display.csv with the subset of users you want to visualize. 
6. Run display.sh. 
7. Go to the address in your browser outputted from the previous step, where you'll have access to the interactive interface.