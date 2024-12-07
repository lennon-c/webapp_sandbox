import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

 

@st.cache_data
def load_data():
    data = pd.read_csv(r"webapp_sandbox/Tweets.csv")
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

@st.cache_data
def data_by_airline(data,airline):
    return data[data['airline']==airline]

@st.cache_data
def data_by_sentiment(data,sentiment):
    return data[data['airline_sentiment']==sentiment]

@st.cache_data
def data_by_hour(data,hour):
    return data[data['tweet_created'].dt.hour == hour]

def get_random_tweet(data,sentiment):
    text = (data_by_sentiment(data, sentiment)
                   .sample(n=1)['text']
                   .iat[0])
    return text

@st.cache_data
def get_sentiment_count(data):
    count = data['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values})
    return count

@st.cache_data
def get_airline_count(data):
    airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
    airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
    return airline_sentiment_count

def sentiments_bar(count=None):
    if count is None:
        count = get_sentiment_count(load_data())
 
    fig = px.bar(count, x='Sentiment', y='Tweets', color='Tweets', height=500)
    st.plotly_chart(fig)

def sentiments_pie(count=None):
    if count is None:
        count = get_sentiment_count(load_data())

    fig = px.pie(count, values='Tweets', names='Sentiment')
    st.plotly_chart(fig)

def mask_from_path(path: str) -> np.ndarray:
    """Loads an image from the given path and returns it as a NumPy array."""
    mask = np.array(Image.open(r"webapp_sandbox/mask.png"))
    return mask

def sentiment_wordcloud(data,sentiment):
    df = data_by_sentiment(data, sentiment)
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() 
                                if 'http' not in word 
                                and not word.startswith('@') 
                                and word != 'RT'])
    wc = WordCloud(stopwords=STOPWORDS,
                           background_color=None,
                           mode= "RGBA",
                           mask=mask_from_path("mask.png"), 
                           prefer_horizontal= 0.75,
                           ) 

    wordcloud = wc.generate(processed_words)
    fig = plt.figure(figsize = (4, 4))
    ax = fig.add_subplot()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_axis_off()
    st.pyplot(fig)

 
def airlines_bar(count = None):
    if count is None:
        count = get_airline_count(load_data())

    fig = px.bar(count, x='Airline', y='Tweets', color='Tweets', height=500)
    st.plotly_chart(fig)

def airlines_pie(count = None):
    if count is None:
        count = get_airline_count(load_data())
    fig = px.pie(count, values='Tweets', names='Airline')
    st.plotly_chart(fig)

def airlines_sentiment_bar(data,selected_airlines):
    fig = make_subplots(rows=1, 
                            cols=len(selected_airlines), 
                            subplot_titles=selected_airlines,
                            shared_yaxes=True,)
    for j, airline in enumerate(selected_airlines):
        count = get_sentiment_count(data_by_airline(data, airline=airline))
        fig.add_trace(
            go.Bar(x=count.Sentiment, y=count.Tweets, showlegend=False),
            row=1, col=j+1
        )

    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)

def airlines_sentiment_pie(data,selected_airlines):
    fig = make_subplots(rows=1, cols=len(selected_airlines), specs=[[{'type':'domain'}]*len(selected_airlines)], subplot_titles=selected_airlines)

    for j, airline in enumerate(selected_airlines):
        count = get_sentiment_count(data_by_airline(data, airline=airline))
        fig.add_trace(
            go.Pie(labels=count.Sentiment, values=count.Tweets, showlegend=True),
            1, j+1
        )
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)


class Widgets:
    about ="""
This application is a **Streamlit dashboard** designed to analyze the sentiments of tweets 🐦.

#### Attribution
- **Code Source**: This application is adapted from the Coursera project _[Create Interactive Dashboards with Streamlit and Python](https://www.coursera.org/projects/interactive-dashboards-streamlit-python)_ by [Snehan Kekre](https://www.coursera.org/instructor/snehan-kekre).
- **Dataset**: The tweet sentiment data is sourced from the _[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)_ dataset, available on [Kaggle](https://www.kaggle.com/).

#### Modifications

This application extends the original project to ensure:

- **Updated Compatibility** with the latest version of Streamlit.
- **Code Restructuring** to improve modularity by separating functionalities into:
    - **Data Handling**: For data loading, filtering, and visualization.
    - **App Widgets**: Defined within a class.
    - **App Logic**: For managing the overall workflow and display.
- **Mask for Word Cloud**.

Modified code on [GitHub - webapp_sandbox](https://github.com/lennon-c/webapp_sandbox). Code : [2_Dashboard.py](https://github.com/lennon-c/webapp_sandbox/blob/434c8ddb4f996d0447835e0d7273a56a6f5e01a5/webapp_sandbox/pages/2_Dashboard.py)
    """
 

    def select_plot(self, key='1'):
        return st.selectbox('Visualization:', 
                            ['None', 'Bar plot', 'Pie chart'],
                            key=key,
                            label_visibility='collapsed',)
    
    def show_raw_data(self):
        return st.checkbox("Show raw data?", 
                           False,
                           )
    def show_random_tweet(self):
        return st.checkbox("Show random tweet?", 
                           False,
                           )
    
    def show_map(self):
        return st.checkbox("Show map?", False)
    
    def show_by_sentiment(self):
        return st.checkbox("Show by sentiment?", False)
    
    def show_by_airline(self):
        return st.checkbox("Show by airline?", False)
    
    def show_by_sentiment_and_airline(self):
        return st.checkbox("Show by sentiment and airline?", False)
    
    def show_word_cloud(self):
        return st.checkbox("Show word cloud?", False)

    def select_sentiment(self):
        return st.radio('Sentiment', 
                        ('positive', 'neutral', 'negative'),
                        # horizontal=True,
                        label_visibility='collapsed',)
    
    def select_airlines(self):
        return st.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'),  label_visibility='collapsed',)
    
    def select_hour(self):
        return st.slider("Hour to look at", 0, 23,  label_visibility='collapsed',)
 
    def box(self, text):
        container = st.container(border=True)
        container.write(text)
        return container
    
 

    
def main():
    ### Setup
    data = load_data()
    display = Widgets()

    ### Sidebar
    with st.sidebar:
        "## Tweets"  
        show_raw_data = display.show_raw_data()
        "#### Select plot type"
        selected_plot = display.select_plot(key='1')
        show_by_sentiment = display.show_by_sentiment()
        show_by_airline = display.show_by_airline()
        show_by_sentiment_and_airline = display.show_by_sentiment_and_airline()

        if show_by_sentiment_and_airline:
            "#### Select airlines"
            selected_airlines = display.select_airlines()
    
        "## By sentiment"
        col1, col2 = st.columns(2, vertical_alignment= "center")
        with col1:
            "##### Select  sentiment"
            sentiment = display.select_sentiment()
        with col2:
            "##### Show"
            show_random_tweet = display.show_random_tweet()
            show_cloud = display.show_word_cloud()
        
        st.divider()
        if show_random_tweet:
            "##### Random tweet "  
            display.box(get_random_tweet(data, sentiment))
        if show_cloud:
            "##### Word cloud" 
            sentiment_wordcloud(data, sentiment)

        "### Tweets map by hour" 
        show_map = display.show_map()
        if show_map:
            "##### Select  time of the day" 
            selected_hour = display.select_hour()


    ### Body
    "# Sentiment Analysis of Tweets about US Airlines"
    if st.button("About"):
        st.info(display.about)
    
    st.divider()
    if show_raw_data:
        data
        st.divider()
    
    if selected_plot  != 'None':
        if show_by_sentiment:
            "## Number of tweets by sentiment"
            if selected_plot == 'Bar plot':
                sentiments_bar()
            elif selected_plot == 'Pie chart':
                sentiments_pie()
        if show_by_airline:
            "## Number of tweets by airline"
            if  selected_plot == 'Bar plot':
                airlines_bar() 
            elif selected_plot == 'Pie chart':
                airlines_pie()
        if show_by_sentiment_and_airline  and selected_airlines and len(selected_airlines) > 0:
            "## Number of tweets by sentiment and airline"
            if selected_plot == 'Bar plot':
                airlines_sentiment_bar(data,selected_airlines)
            elif selected_plot == 'Pie chart':
                airlines_sentiment_pie(data, selected_airlines)
    
    if show_map:
        st.divider()
        "### Tweet locations based on time of the day"
        map_data = data_by_hour(data,hour=selected_hour)
        f'{len(map_data)} tweets between {selected_hour}:00 and {selected_hour + 1}:00'
        st.map(map_data)
 

if __name__ == '__main__':
    main()
 