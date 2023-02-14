'''
Code: Scrape Twitter given a set of user names. Place the results in a csv file and output it. 
Author: Tyler J Burns, PhD
Date: November 2, 2022
Purpose: To ultimately analyze full archives of users using NLP and whatever other tools are relvant. 
'''
import sys
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import glob

# from https://github.com/mehranshakarami/AI_Spectrum/blob/main/2022/snscrape/tweets.py
# Don't forget to caffeinate: caffeinate -is python3 scrape_tweets.py

def get_tweets(search_query, max_tweets, last_date, first_date):
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
    query = search_query + ' until:' + last_date + ' since:' + first_date + ' -filter:retweets'

    tweets = []

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


# Get the tweets
now = datetime.now()
now = now.strftime('%Y-%m-%d')

search = sys.argv[1]
max_tweets = int(sys.argv[2])
print(search)
df = get_tweets(search, max_tweets, now, '2000-01-01')
df = df.sort_values(by=['Likes'], ascending=False)
df.to_csv('output/' + search + '.csv', index=False)


