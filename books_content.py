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


def compute_cosine_similarity_matrix(data, method='tfidf'):
    '''Compute the cosine similarity matrix based on the chosen method'''
    if method not in ['tfidf', 'count']:
        raise ValueError("Invalid method. Choose either 'tfidf' or 'count'.")
        
    vectorizer = TfidfVectorizer(stop_words='english') if method == 'tfidf' else CountVectorizer(stop_words='english')
    try:
        matrix = vectorizer.fit_transform(data)
    except Exception as e:
        raise ValueError(f"Error during vectorization: {e}")

    cosine_sim_matrix = linear_kernel(matrix, matrix) if method == 'tfidf' else cosine_similarity(matrix, matrix)
    return cosine_sim_matrix
        


def get_recommendations(title, method='tfidf'):
    """Get book recommendations based on the chosen method"""
    df = read_data(URL)
    if method == 'tfidf':
        cosine_sim_matrix = compute_cosine_similarity_matrix(df['description'])
    elif method == 'count':
        cosine_sim_matrix = compute_cosine_similarity_matrix(df['generes'], method='count')
    else:
        raise ValueError("Invalid method. Please choose 'tfidf' or 'count'.")
    
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    index = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices]


if __name__ == '__main__':
    title = input('Write your book name:\n')
    print("Recommendations based on TF-IDF:")
    print(get_recommendations(title, method='tfidf'))
    print("Recommendations based on Count Vectorizer:")
    print(get_recommendations(title, method='count'))