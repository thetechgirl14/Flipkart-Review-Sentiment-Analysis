#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))

data_path = "flipkart_reviews.csv" #Path to CSV


# Starting review sentiment analysis by importing the necessary Python libraries and the dataset:

def load_the_dataset():
    data = pd.read_csv(data_path)
    return data


# As this dataset is very large, it contains some missing values, remove all the rows containing the missing values

def pre_processing():
    data = load_the_dataset()
    data.dropna(inplace=True)
    return data

# The Rating column of this dataset contains the ratings that customers have given to the product based on their experience with the product.
# Take a look at the rating breakdown to see how most customers rate the products they buy from Flipkart:

def distribution_of_ratings():
    data = pre_processing()
    ratings = data['Rating'].value_counts().sort_values(ascending=False)

    return ratings


# Now let’s see how most people rated the products they bought from Flipkart:

def data_visualization():
    ratings = distribution_of_ratings()

    colors = ['#0099ff', '#00cc99', '#ffcc00', '#ff6666', '#cc99ff']

    plt.figure(figsize=(10, 6))
    bars = plt.barh(ratings.index, ratings.values, color=colors)

    plt.title('Rating Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height() / 2, str(int(width)), color='black', ha='center', va='center')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.gca().xaxis.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f2f2f2')

    plt.tight_layout()

    plt.savefig("Rating_Distribution.png")
    plt.show()


# Adding three more columns to this dataset as Positive, Negative, and Neutral by calculating the sentiment scores of the customer reviews mentioned in the Text column of the dataset:

def add_new_columns():
    # nltk.download('vader_lexicon')
    data = pre_processing()
    sia = SentimentIntensityAnalyzer()
    data['Positive'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['pos'])
    data['Negative'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['neg'])
    data['Neutral'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['neu'])
    data = data[['Review', 'Positive', 'Negative', 'Neutral']]
    return data


# Now see how most people rated the products they bought from Flipcart:
# Then see the total of all sentiment scores:
# return the sum of positive, negative and neutral reviews respectively(Set to 2 decimal places)

def sentiment_scores():
    data = add_new_columns()
    x = round(data['Positive'].sum(), 2)
    y = round(data['Negative'].sum(), 2)
    z = round(data['Neutral'].sum(), 2)
    return x, y, z

# Visualization of sentiment scores
def sentiment_visualization():
    x, y, z = sentiment_scores()

    # Pie chart for sentiment scores
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [x, y, z]
    colors = ['green', 'red', 'blue']
    explode = (0.1, 0, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Scores')
    plt.savefig("Sentiment_Scores.png")
    plt.show()

if __name__ == '__main__':


    #ratings = distribution_of_ratings()
    #print("Rating Breakdown:")
    #print(ratings)

    data_visualization()

    #x, y, z = sentiment_scores()
    #print("Sum of Sentiment Scores:")
    #print("Positive:", x)
    #print("Negative:", y)
    #print("Neutral:", z)

    sentiment_visualization()
