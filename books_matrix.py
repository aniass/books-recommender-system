'''The recommendation system with matrix factorization method'''

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


URL = '\Datasets\goodreads_book\books.csv'

popularity_threshold = 1000


def read_data(path):
    '''Read the data'''
    df = pd.read_csv(path, error_bad_lines=False)
    return df


def preparing_data():
    '''Preparing data to modelling'''
    df = read_data(URL)
    id_rating = df[['bookID', 'average_rating', 'title', 'ratings_count']]
    # choose books which received above 1000 ratings
    id_rating2 = id_rating.query('ratings_count >= @popularity_threshold')
    # convert the table into a 2D matrix
    id_rating_pivot = id_rating2.pivot(index='bookID', columns='title', values='average_rating').fillna(0)
    return id_rating_pivot


def dimensionality_reduction(id_rating_pivot):
    '''Calculating model and dimensionality reduction'''
    # transpose the matrix 
    X = id_rating_pivot.values.T
    # SVD model
    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix = SVD.fit_transform(X)
    # calculating the Pearson’s R correlation coefficient 
    corr = np.corrcoef(matrix)
    return corr
    

def matrix_factorization(name):
    """The function enables to find the books that have high correlation coefficients 
    (between 0.9 and 1.0) with chosen book and get the recommendations for it."""
    id_rating_pivot = preparing_data()
    corr = dimensionality_reduction(id_rating_pivot)
    book_title = id_rating_pivot.columns
    book_list = list(book_title)
    book = book_list.index(name)
    corr_book = corr[book]
    print("Recommended books are:\n")
    print(list(book_title[(corr_book < 1.0) & (corr_book > 0.9)]))
    
    
if __name__ == '__main__':
    matrix_factorization("Animal Farm")
    matrix_factorization("The Da Vinci Code")
    