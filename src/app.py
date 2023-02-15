'''
Description: his creates the web app. It runs Dash on a local server. 
The user runs this script after the dimension reduction and clustering has been run on the orignal tweet metadata file. 
Author: Tyler J Burns
Date: November 2, 2022
'''

from dash import Dash, dcc, html, dash_table, Input, Output, callback_context, State
import plotly.express as px
import pandas as pd
import json
import re
import numpy as np
from datetime import datetime

# You have to include these two lines if you want it to run in Heroku
app = Dash(__name__)
server = app.server


def search_bar(input, text):
    '''
    Takes a string with or without logical values (AND, OR) as input, runs that on another given string, and returns boolean corresponding to whether the input was in the other string.

    Args:
        input: The search term. 
        text: The text it will be doing the search on.
    Returns:
        boolean value correspoding to whether the input was in the text

    Note: 
        For logic-based search, you can add (all caps) AND or OR. But you can't add both of them. 

    Example:
        search_bar('beer OR wine", "This beer is good") returns True.
        search_bar('beer AND wine", "This beer is good") returns False.
        search_bar('beer AND wine OR cheese', 'This beer is good') returns False, because this function cannot use combos of AND and OR
    '''
    text = text.lower()
    bool_choice = [input.find('AND') != -1, input.find('OR') != -1]
    
    if(sum(bool_choice) == 0):
        result = text.find(input.lower()) != -1
        return(result)
    
    if sum(bool_choice) == 2:
        return(False)
    if bool_choice[0]:
        bool_choice = 'AND'
    elif bool_choice[1]:
        bool_choice = 'OR'

    input = input.split(' ' + bool_choice + ' ')
    input = [i.lower() for i in input]

    if bool_choice  == 'AND':
        result = [all(text.find(i.lower()) != -1 for i in input)]
    elif bool_choice == 'OR':
        result = [any(text.find(i.lower()) != -1 for i in input)]

    return(result[0])

# tmp.csv is the tweet metadata + dimension reduction + clustering script
df = pd.read_csv('src/tmp.csv', lineterminator='\n')

# Dev
# df = df[df['sentiment'] > 0.8]

# This is for the per-user dropdown menu initialized later in the code
users = list(set(df['User']))
users.append('all_users')

# NULL or NaN gets added to the rows sometimes upstream. Have not figured out why.
df = df[df['Tweet'].notnull()] 
df = df[df['Date'].notnull()] 

# NOTE some of the tweets are repeats. We need to get rid of these. 

# We have the url in markdown format, so it only shows up as a hyperlink
df['Url'] = ['[Go to tweet]' + '(' + i + ')' for i in df['Url']]

# Tweets often have links included in the text at the end. We get rid of them here.
df['Tweet'] = [i.split('http')[0] for i in df['Tweet']]
df['Tweet'] = df['Tweet'].str.wrap(30)
df['Tweet'] = df['Tweet'].apply(lambda x: x.replace('\n', '<br>')) # bug
df['Year'] = [i.split('-')[0] for i in df['Date']]

# There are some complexities in terms of handling "date" objects that I take care of here
# We create the time delta so we can specify tweets in terms of how recent or by year
df['Date'] = [i.replace('T', ' ') for i in df['Date']]
df['Date'] = [i.replace('Z', '') for i in df['Date']] 
df['Date'] = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in df['Date']]
df['Time_delta'] = [df['Date'][0] - i for i in df['Date']]
df['Time_delta'] = [i.days for i in df['Time_delta']]
df['Date'] = [str(i) for i in df['Date']]

# We include now so when the year changes, the interface updates
now = datetime.now()
curr_year = now.year

# We get rid of unnecessary columns for downstream processing
df_sub = df[['Url', 'User', 'Date', 'Likes', 'Tweet']]

# This is the layout of the page. We are using various objects available within Dash. 
fig = px.scatter() # The map
fig2 = px.bar() # The yearly trends bar plot
app.layout = html.Div([
    html.A(html.P('Click here for instructions'), href="https://tjburns08.github.io/app_instructions.html"),
    dcc.Dropdown(users, id='user-dropdown', value = users[0]),
    dcc.Dropdown(['Today', 'Last 7 days', 'All years'] + list(range(curr_year, 2006, -1)), id='year-dropdown', value='Last 7 days'),
    dcc.Textarea(
        placeholder='Type keywords separated by AND or OR',
        id = 'user-input',
        value='',
        style={'width': '100%'}
    ),
    html.Button('Submit', id='value-enter'),
    dcc.Graph(
        id='news-map',
        figure=fig
    ),
    html.Plaintext('Info of tweet you clicked on.'),
    dash_table.DataTable(data = df_sub.to_dict('records'), style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    }, id='news-table', fill_width = False, columns=[{'id': x, 'name': x, 'presentation': 'markdown'} if x == 'Url' else {'id': x, 'name': x} for x in df_sub.columns]),

    dcc.Graph(
        id = 'news-trends',
        figure=fig2
    ),

    # Dev
    html.Plaintext('Top tweets given search term, or top tweets of all time if no search term.'),
    dash_table.DataTable(data = df_sub.to_dict('records'), style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    }, id='top-table', fill_width = False, columns=[{'id': x, 'name': x, 'presentation': 'markdown'} if x == 'Url' else {'id': x, 'name': x} for x in df_sub.columns])

])

# This allows the user to click on a point on the map and get a single entry corresponding to that article
# TODO consider returning the article and its KNN
@app.callback(
    Output('news-table', 'data'),
    Input('news-map', 'clickData'))

def click(clickData):
    if not clickData:
        return
    tweet = clickData['points'][0]['customdata'][2]
    filtered_df = df_sub[df_sub['Tweet'] == tweet]
    filtered_df['Tweet'] = [re.sub('<br>', ' ', i) for i in filtered_df['Tweet']]
    filtered_df['Tweet'] = [i.split('https:')[0] for i in filtered_df['Tweet']]
    return filtered_df.to_dict('records')

# This updates the "top tweets" table every time the year dropdown changes
@app.callback(
    Output('top-table', 'data'),
    Input('value-enter', 'n_clicks'),
    Input('year-dropdown', 'value'),
    State('user-input', 'value'))

def update_table(n_clicks, year_value, input_value):
    
    # TODO do we put the "df" object itself as input to this function?
    # TODO filtered_df = df_sub?
    filtered_df = df # Make local variable. There might be a less ugly way to do this.

    if(year_value != 'All years'):
        # Note that this assumes an update the day of
        # TODO get the wording right here, to account for not always updating
        if(year_value == 'Last 7 days'):
            filtered_df = filtered_df[filtered_df['Time_delta'] < 8]
        elif(year_value == 'Today'):
            filtered_df = filtered_df[filtered_df['Time_delta'] == 0]
        else:
            filtered_df = filtered_df[filtered_df['Year'] == str(year_value)]
    
    if filtered_df.shape[0] == 0:
        return
    
    rel_rows = []
    for i in filtered_df['Tweet']:
        rel_rows.append(search_bar(input_value, i))

    # Re-initialize df_sub given our df has been filtered above
    # TODO to this in a less ugly way please
    filtered_df_sub = filtered_df[['Url', 'User', 'Date', 'Likes', 'Tweet']]
    filtered_df_sub = filtered_df_sub[rel_rows]

    filtered_df_sub['Tweet'] = [re.sub('<br>', ' ', i) for i in filtered_df_sub['Tweet']]
    filtered_df_sub['Tweet'] = [i.split('https:')[0] for i in filtered_df_sub['Tweet']]
    filtered_df_sub = filtered_df_sub.sort_values('Likes', ascending=False)
    return filtered_df_sub.to_dict('records')

# This updates the map given the dropdowns and the value entered into the search bar
# TODO change news-map
@app.callback(  
    Output('news-map', 'figure'),
    Input('user-dropdown', 'value'), 
    Input('year-dropdown', 'value'),
    Input('value-enter', 'n_clicks'),
    State('user-input', 'value'))

def update_plot(source_value, year_value, n_clicks, input_value):
    user_context = callback_context.triggered[0]['prop_id'].split('.')[0]

    tmp = df
    if(source_value != 'all_users'):
        tmp = df[df['User'] == source_value]

    if(year_value != 'All years'):
        # Note that this assumes an update the day of
        # TODO get the wording right here, to account for not always updating
        if(year_value == 'Last 7 days'):
            tmp = tmp[tmp['Time_delta'] < 8]
        elif(year_value == 'Today'):
            tmp = tmp[tmp['Time_delta'] == 0]
        else:
            tmp = tmp[tmp['Year'] == str(year_value)]

    if(user_context == 'value-enter' or input_value != ''):
        # input_values = input_value.lower().split(',')
        # Keyword logic
        # TODO make this a standalone function
        rel_rows = []

        # TODO add OR to this
        for i in tmp['Tweet']:
            # rel_rows.append(all([v in i.lower() for v in input_values]))
            rel_rows.append(search_bar(input_value, i))
        tmp = tmp[rel_rows]

    if(source_value == 'all_users'):
        fig = px.scatter(tmp, x = 'umap1', y = 'umap2', hover_data = ['Date', 'Likes', 'Tweet', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'], color = 'User', size = 'Likes', size_max = 50, title = 'Compare user mode')
    else:
        fig = px.scatter(tmp, x = 'umap1', y = 'umap2', hover_data = ['Date', 'Likes', 'Tweet', 'keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5'], color = 'sentiment_label', size = 'Likes', size_max = 50, title = 'Context similarity map of tweets')
    
    # DarkSlateGrey
    fig.update_traces(marker=dict(line=dict(width=0.1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return(fig)


# This updates the "yearly trends" bar plot
# NOTE it might be better to do this elsewhere, like in another app because the dropdown by year might make it confusing
# Or, we just have this callback independent of the year dropdown
@app.callback(  
    Output('news-trends', 'figure'),
    Input('user-dropdown', 'value'), 
    Input('value-enter', 'n_clicks'),
    State('user-input', 'value'))

def update_plot2(source_value, n_clicks, input_value):
    user_context = callback_context.triggered[0]['prop_id'].split('.')[0]
    tmp = df
    if(source_value != 'all_users'):
        tmp = df[df['User'] == source_value]
    year_counts = tmp['Year'].value_counts(sort = False).rename_axis('Year').reset_index(name = 'TotalTweets').sort_values('Year')
    # print(year_counts)
    
    # TODO use this to normalize
    if(user_context == 'value-enter' or input_value != ''):
        # input_values = input_value.lower().split(',')
        # Keyword logic
        rel_rows = []
        for i in tmp['Tweet']:
            # rel_rows.append(all([v in i.lower() for v in input_values]))
            rel_rows.append(search_bar(input_value, i))
        tmp = tmp[rel_rows]

    # Tabulate the year
    # NOTE ugly bug around having duplicate column names with different values
    year_counts2 = tmp['Year'].value_counts(sort = False).rename_axis('year').reset_index(name = 'TweetSubset').sort_values('year')
    #year_counts2['Year'] = year_counts2['year']
    # print(year_counts2)
    year_counts2 = pd.concat([year_counts, year_counts2], axis = 1).fillna(0)
    year_counts2 = year_counts2.drop(['year'], axis = 1)
    year_counts2['TweetPct'] = 100*(year_counts2['TweetSubset']/year_counts2['TotalTweets'])

    # print(year_counts2)
    fig2 = px.bar(year_counts2, x='Year', y='TweetPct', title = 'Search term percent of total tweets')

    # DarkSlateGrey
    fig2.update_traces(marker=dict(line=dict(width=0.1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return(fig2)

if __name__ == '__main__':
    app.run_server(debug=True)