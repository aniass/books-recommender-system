'''The simple recommendation system by Weighted Rating method'''

import pandas as pd


URL = r'Datasets\google_books\google_books_1299.csv'


def clean_data(data):
    '''Function to clean data'''
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
    """Function to calculate a simple recommendations based on Weighted Rating"""
    C = df['rating'].mean()
    m = df['voters'].quantile(0.90)
    V = df['voters']
    R = df['rating']
    
    # Filter out qualified books into a new DataFrame
    q_books = df[df['voters'] >= m].copy()
    
    # Calculate the weighted rating
    q_books['score'] = (V/(V+m) * R) + (m/(m+V) * C)
    
    # Sort the DataFrame by 'score' in descending order
    q_books = q_books.sort_values('score', ascending=False)
    
    # Display the top 10 books
    print(q_books[['title', 'voters', 'rating', 'score']].head(10))


if __name__ == '__main__':
    data = read_data(URL)
    calculate_weighted_rating(data)
