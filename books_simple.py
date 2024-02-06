'''The simple recommendation system by Weighted Rating method'''

import pandas as pd

# Constants
URL = r'Datasets\google_books\google_books_1299.csv'
MIN_VOTERS_PERCENTILE = 0.90
TOP_N_BOOKS = 10


def clean_data(data):
    '''Clean and preprocess data'''
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # Remove duplicates based on 'title'
    df = data.drop_duplicates(keep=False)
    df = df.drop_duplicates(['title'], keep='first')
    # Remove commas, filling missing with 0 and convert to integer in the 'voters' column
    df['voters'] = df['voters'].replace(',','', regex=True).fillna(0).astype(int)
    return df
    

def read_data(path):
    '''Read and preprocess data'''
    data = pd.read_csv(path)
    df = clean_data(data)
    return df


def calculate_weighted_rating(df):
    """Calculate and print recommendations based on Weighted Rating"""
    C = df['rating'].mean()
    m = df['voters'].quantile(MIN_VOTERS_PERCENTILE)
    V = df['voters']
    R = df['rating']
    
    qualified_books = df[df['voters'] >= m].copy()
    # Calculate the weighted rating
    qualified_books['score'] = (V/(V+m) * R) + (m/(m+V) * C)
    qualified_books = qualified_books.sort_values('score', ascending=False)
    
    print("Top {} Books based on Weighted Rating:".format(TOP_N_BOOKS))
    print(qualified_books[['title', 'voters', 'rating', 'score']].head(TOP_N_BOOKS))


if __name__ == '__main__':
    data = read_data(URL)
    calculate_weighted_rating(data)
