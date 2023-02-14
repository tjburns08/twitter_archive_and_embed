'''
Code: Scrape Twitter given a set of user names. Place the results in a csv file and output it. 
Author: Tyler J Burns, PhD
Date: November 2, 2022
Purpose: To ultimately analyze full archives of users using NLP and whatever other tools are relvant. 
'''
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import glob

# from https://github.com/mehranshakarami/AI_Spectrum/blob/main/2022/snscrape/tweets.py
# Don't forget to caffeinate: caffeinate -is python3 scrape_tweets.py

def get_tweets(user, max_tweets, last_date, first_date):
    '''
    Description: Given a user name and a date range, scrape the tweets. 
    Params: 
        max_tweets: the maximum number of tweets to pull. 
        user: the twitter handle, a string without the @
        last_date: the most recent date to scrape.
        firrst_date: the initial date to scrape. 
    Return: DataFrame of tweets and tweet metadata
    Note: Date objects are derived using X.strftime('%Y-%m-%d')
    '''
    # query = "(from:cnnbrk) until:2022-09-13 since:2007-01-01"
    # query = '(from:' + user + ') until:2022-09-13 since:2007-01-01'
    query = '(from:' + user + ') until:' + last_date + ' since:' + first_date
    tweets = []
    limit = max_tweets

    count = 0
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        # print(vars(tweet))
        # break
        if len(tweets) == max_tweets:
            break
        else:
            tweets.append([tweet.url, tweet.date, tweet.id, tweet.conversationId, tweet.lang, tweet.source, tweet.username, tweet.likeCount, tweet.retweetCount, tweet.replyCount, tweet.quoteCount, tweet.content, tweet.renderedContent, tweet.media, tweet.outlinks, tweet.tcooutlinks, tweet.retweetedTweet, tweet.quotedTweet, tweet.mentionedUsers])
    
        count += 1
        if(count % 1000 == 0):
            print(str(count) + ' tweets ' + str(tweet.date))
        
    df = pd.DataFrame(tweets, columns=['Url', 'Date', 'ID', 'ConversationID', 'Language', 'Source', 'User', 'Likes', 'Retweets', 'Replies', 'Quotes', 'Tweet', 'RenderedTweet', 'Media', 'Outlinks', 'TCooutlinks', 'RetweetedTweet', 'QuotedTweet', 'MentionedUsers'])
    return df

# A list of twitter handles
users = pd.read_csv('users.csv')
users = users['user'].to_list()

# Here are the existing csv files, previously outputted from this script. The code below adds to them. 
curr_files = glob.glob('output/*')
now = datetime.now()
now = now.strftime('%Y-%m-%d')

for i in users:
    print(i)

    if any([i + '_' in f for f in curr_files]):
        print('User in curr_files')

        # Read in file
        df_orig = pd.read_csv('output/' + i + '_tweets_scraped.csv', lineterminator='\n')
        print('df_orig')
        print(df_orig)

        if df_orig.shape[0] == 0:
            continue

        # Get most recent date
        first = parse(df_orig['Date'][0])
        first = first.strftime('%Y-%m-%d')
        print('first')
        print(first)

        # Scrape tweets until that date, inclusive (to make sure we didn't miss any times)
        df = get_tweets(i, 10**9, now, first)
        print('scraped_tweets')
        print(df)
       
        df = df.append(df_orig, ignore_index = True)
        # df = pd.concat([df,df_orig]).drop_duplicates().reset_index(drop=True)
        print('final_merged_df')
        print(df)

    else: 
        print('User not detected in database')   
        first = '2000-01-01'
        df = get_tweets(i, 10**9, now, first) 
    
    df.to_csv('data/scraped_tweets/' + i + '_tweets_scraped.csv', encoding='utf-8-sig', index=False)