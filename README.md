
# Books Recommendation System

## General info
The project concerns the books recommendation system. It includes data analysis, data preparation and explored three kind of recommendations - the simplest recommendations, content-based filtering and collaborative filtering (KNN model and matrix factorization). The final result will show that the user can input one book's name or author then the system can provide the other most possible books that he can to read.

### Dataset
In our analysis we used two datasets come from Kaggle: [Goodreads books](https://www.kaggle.com/jealousleopard/goodreadsbooks) and [Google Books](https://www.kaggle.com/bilalyussef/google-books-dataset). 

## Motivation
The Recommendation System is based on previous (past) behaviours, it predicts the likelihood that a user would prefer an item. Many of applications for example Netflix uses recommendation system. It suggests people new movies according to their past activities that are like watching and voting movies. The purpose of a book recommendation system is to predict buyer’s interest and recommend books to them respectively. That system can take into regard many parameters like book content and quality by filtering user reviews. 

## Project contains:
- Books recommendation system by using simplest methods and content-based filtering - **books_content.ipynb**
- Books recommendation system by using collaborative filtering - **books_colaborative_filtering.ipynb**
- Python script with the simple recommendation system - **books_simple.py** 
- Python script with the recommendation system by content-based filtering method - **books_content.py**  
- Python script with the recommendation system by collaborative filtering method - **books_cf.py**  
- Python script with the recommendation system by matrix factorization method - **books_matrix.py**  


## Summary
The main aim of this project was build book recommendation system. Based on analysis of two datasets (Goodreads books and Google Books) I have explored three kind of recommendations: the simplest recommendations, content-based filtering and collaborative filtering (KNN model and matrix factorization). I have started with data analysis to better meet the data. Then I have cleaned data and prepared them to the modelling.

In the first approach I have used two methods: the simplest recommendations and content-based filtering. Simple Recommendations are the basic systems that recommend the best items based on a specific metric or score. In this method calculeted weighted rating which takes into account the average rating and number of votes it has collected. After the calculation I have get python function that user can input one book's name then the system can provide the other most possible books that he can to read. The content-based recommendations suggest similar items based on a particular item. This method uses item data such as a description, genre etc. In this point I used Term Frequency-Inverse Document Frequency (TF-IDF) method to find the similarity between books. I have also created recommendation function which takes the book title as input and the outputs are the similar books.

In the second part I have used a method colaborative filtering and matrix factorization. Collaborative filtering method builds a model from a user’s past behaviors as well as similar decisions made by other users. Then this model is used to predict items (or ratings for items) that the user may have an interest in. To apply an item based collaborative filtering I have used a KNN algorithm. Matrix factorization is a collaborative filtering method to find the relationship between items’ and users’ entities. One of the Matrix Factorization models for identifying latent factors is singular value decomposition (SVD)  and I have used it for the analysis. Finally I have created functions for recommendations based on these two methods where that user can input one book's name then the system can provide the other most possible books that he can to read. I have also created function for recommendations by author where I can received recommended books after input author's name. As one can see the models show a pretty decent results.

## Technologies

The project is created with:
- Python 3.6
- libraries: pandas, numpy, sklearn, seaborn, matplotlib.

**Running the project:**

To run this project use Jupyter Notebook or Google Colab.

You can run the scripts in the terminal:

    books_simple.py
    books_content.py
    books_cf.py
    books_matrix.py

