'''The recommendation system with simplest recommendations method'''

import pandas as pd

URL = 'C:\Python Scripts\Datasets\google_books\google_books_1299.csv'


def clean_data(data):
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    # removing duplicates
    df = data.drop_duplicates(keep=False)
    df = df.drop_duplicates(['title'], keep='first')
    # removing commas
    df['voters'] = df['voters'].replace(',','', regex=True)
    # filling with 0
    df['voters'] = df['voters'].fillna(0)
    # changing type for int
    df['voters'] = df['voters'].astype(int)
    return df
    

def read_data(path):
    data = pd.read_csv(path)
    df = clean_data(data)
    return df


def simple_recommend(df):
    C = df['rating'].mean()
    m = df['voters'].quantile(0.90)
    V = df['voters']
    R = df['rating']
    # filtering out all qualified books into a new DataFrame
    q_book = df.copy().loc[df['voters'] >= m]
    q_book['score'] = (V/(V+m) * R) + (m/(m+V) * C)
    q_book = q_book.sort_values('score', ascending=False)
    # the top 15 movies
    print(q_book[['title', 'voters', 'rating', 'score']].head(10))


if __name__ == '__main__':
    data = read_data(URL)
    simple_recommend(data)
