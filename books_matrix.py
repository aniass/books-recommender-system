'''The recommendation system with matrix factorization method'''

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import warnings

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
URL = r'Datasets/goodreads book/books.csv'
POPULARITY_THRESHOLD = 1000
NUM_COMPONENTS = 12
CORRELATION_THRESHOLD_MIN = 0.9
CORRELATION_THRESHOLD_MAX = 1.0


def read_data(path):
    '''Read the data'''
    df = pd.read_csv(path, error_bad_lines=False)
    return df


def prepare_data(path):
    '''Preparing data to modelling'''
    df = read_data(path)
    id_rating = df[['bookID', 'average_rating', 'title', 'ratings_count']]
    # choose books which received above 1000 ratings
    id_rating = id_rating[id_rating['ratings_count'] >= POPULARITY_THRESHOLD]
    # convert the table into a 2D matrix
    id_rating_pivot = id_rating.pivot(index='bookID', columns='title', values='average_rating').fillna(0)
    return id_rating_pivot


def dimensionality_reduction(id_rating_pivot):
    '''Calculating model and dimensionality reduction'''
    # transpose the matrix 
    X = id_rating_pivot.values.T
    # SVD model
    SVD = TruncatedSVD(n_components=NUM_COMPONENTS, random_state=17)
    matrix = SVD.fit_transform(X)
    # calculating the Pearson’s R correlation coefficient 
    corr = np.corrcoef(matrix)
    return corr
    

def matrix_factorization(name, data_path):
    """The function enables to find the books that have high correlation coefficients 
    (between 0.9 and 1.0) with chosen book and get the recommendations for it."""
    id_rating_pivot = prepare_data(data_path)
    corr = dimensionality_reduction(id_rating_pivot)
    book_title = id_rating_pivot.columns
    book_list = list(book_title)
    book = book_list.index(name)
    corr_book = corr[book]
    print("Recommended books are:\n")
    print(list(book_title[(corr_book < CORRELATION_THRESHOLD_MAX) & (corr_book > CORRELATION_THRESHOLD_MIN)]))
   
    
if __name__ == '__main__':
    matrix_factorization("Animal Farm", URL)
    matrix_factorization("The Da Vinci Code", URL)
    