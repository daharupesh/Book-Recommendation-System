# Book Recommender System Using Machine Learning

## Overview

This is a web application developed with Streamlit that recommends books to users based on a machine learning model. The recommender system uses a collaborative filtering approach to suggest books to users based on their preferences and ratings.

## Features

- **Book Selection**: Users can either type in the name of a book or select from a dropdown list of available books.
- **Recommendation**: After selecting a book and clicking the 'Recommend' button, the system provides a list of 10 recommended books along with their cover images.
- **Visual Interface**: The recommended books are displayed in a 5-column grid layout for better visualization.

## Data and Model

The system uses the following pre-trained machine learning model and data stored in pickle files:

- **Model**: `model.pkl` - The trained k-nearest neighbors (KNN) model for book recommendation.
- **Books Data**: `books_name.pkl` - A list of book names.
- **Rating Data**: `final_rating.pkl` - A dataset containing book titles, ratings, and cover image URLs.
- **Pivot Table**: `book_pivot.pkl` - A pivot table containing user-book ratings.

## How to Run the App

To run the app locally, follow these steps:

1. Clone the repository.
2. Install the required packages:
    ```bash
    pip install streamlit numpy
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4. Select a book from the dropdown or type in a book name.
5. Click the 'Recommend' button to view the recommended books.

## Code Overview

### Data Loading

```python
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/books_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
