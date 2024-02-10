'''The recommendation system with content-based filtering'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Constants
URL = r'Datasets/google_books/google_books_1299.csv'


def clean_data(data):
    '''Clean and preprocess the data'''
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = data.drop_duplicates(keep=False)
    df = df.drop_duplicates(['title'], keep='first')
    df['description'] = df['description'].fillna('')
    return df
    

def read_data(path):
    '''Read and preprocess data'''
    data = pd.read_csv(path)
    df = clean_data(data)
    return df


'''The recommender system based on descriptions of books by TF-IDF method'''


def cosine_sim(df):
    '''Function to computing the cosine similarity matrix'''
    # creating the TF-IDF Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine 


def get_recommendations(title):
    """Function takes the book title as an input and outputs are the similar books"""
    df = read_data(URL)
    cosine_sim1 = cosine_sim(df)
    # reverse map of indices and book titles:
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    # creating recommendations
    index = indices[title]
    sim_score = list(enumerate(cosine_sim1[index]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:11]
    book_indices = [i[0] for i in sim_score]
    return df['title'].iloc[book_indices]


'''The recommender system based on genres of books by Count Vectorizer method'''
 

def cosine_sim2(df):
    '''Function to computing the cosine similarity matrix'''
    # define a Count Vectorizer object
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['generes'])
    # cosine similarity matrix based on the count matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim
    

def get_recommendations2(title):
    """Function takes the book title as an input and outputs are the similar books"""
    df = read_data(URL)
    cosine_sim1 = cosine_sim2(df)
    # resetting index of main DataFrame and constructing reverse mapping
    df3 = df.reset_index()
    indices = pd.Series(df3.index, index=df3['title'])
    # creating recommendations
    index = indices[title]
    sim_score = list(enumerate(cosine_sim1[index]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:11]
    book_indices = [i[0] for i in sim_score]
    return df['title'].iloc[book_indices]


if __name__ == '__main__':
    title = input('Write your book name:\n')
    print(get_recommendations(title))
    print(get_recommendations2(title))
