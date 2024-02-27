'''The recommendation system with Collaborative Filtering by KNN model'''

import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler


URL = r'C:\Python Scripts\Datasets\goodreads_book\books.csv'

LANGUAGE_MAP = {
    'eng': 'English', 'en-US': 'English', 'en-GB': 'English', 'enm': 'English', 'en-CA': 'English',
    'fre': 'French', 'spa': 'Spanish', 'mul': 'Multiple language',
    'grc': 'Greek', 'ger': 'German', 'jpn': 'Japanese', 'ara': 'Arabic',
    'nl': 'Dutch', 'zho': 'Chinese', 'lat': 'Latvian', 'por': 'Portuguese',
    'srp': 'Serbian', 'ita': 'Initial teaching language', 'rus': 'Russian', 'msa': 'Modern Standard Arabic',
    'glg': 'Galician', 'wel': 'Welsh', 'swe': 'Swedish', 'tur': 'Murik', 'gla': 'Turkish', 'ale': 'Gaelic'
}


def clean_data(data):
    '''Function to clean data'''
    data['language_code'] = data['language_code'].map(LANGUAGE_MAP)
    return data


def read_data(path):
    '''Read and clean data'''
    data = pd.read_csv(path, error_bad_lines=False)
    df = clean_data(data) 
    return df


def create_rating(row):
    """The function to create a column rating between"""
    if row >= 0 and row <=1:
        return '0-1'
    if row >= 1 and row <=2:
        return '1-2'
    if row >= 2 and row <=3:
        return '2-3'
    if row >= 3 and row <=4:
        return '3-4'
    if row >= 4 and row <=5:
        return '4-5'


def preprocess_data(df):
    '''Data preparation for modeling'''
    df['rating_between'] = df['average_rating'].apply(create_rating)
    # createing two new DataFrames to assign a values 1 and 0 for chosen columns
    rating_df = pd.get_dummies(df['rating_between'])
    language_df = pd.get_dummies(df['language_code'])
    # connection the two data frames into one 
    features = pd.concat([
            rating_df, 
            language_df, 
            df['average_rating'], 
            df['ratings_count']
    ], axis=1)
    return features


def get_model(features, n_neighbors=6):
    '''Calculating the model'''
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(features)
    # calculating KNN model
    model = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    model.fit(feature)
    distance, indices = model.kneighbors(feature)
    return distance, indices 


def make_recommendations(name):
    """The function to get recommendations by name"""
    df = read_data(URL)
    features = preprocess_data(df)
    distance, indices = get_model(features)
    book_list = []
    book_id = df[df['title'] == name].index
    book_id = book_id[0]
    for newid in indices[book_id]:
        book_list.append(df.loc[newid].title)
    print("Recommendations by the book name:\n")
    for i in range(0,len(book_list)):
        print(f"{i+1}){book_list[i]}")
    

def author_recommendations(author):
    """The function to get recommendations by the author"""
    df = read_data(URL)
    features = preprocess_data(df)
    distance, indices = get_model(features)
    author_list = []
    books=[]
    author_id = df[df['authors'] == author].index
    author_id = author_id[0]
    for newid in indices[author_id]:
        author_list.append(df.loc[newid].authors)
        books.append(df.loc[newid].title)
    print("Based the author recommended books are:\n")    
    for i in range(0,len(author_list)):
      print(f"{i+1})Author:{author_list[i]}, '{books[i]}'")


if __name__ == '__main__':
    make_recommendations("Jane Eyre")
    author_recommendations("Dan Brown")
