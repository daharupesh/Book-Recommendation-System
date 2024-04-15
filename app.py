import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title='Book Recommender System', page_icon=':books:', layout='wide')

st.title('Book Recommender System Using Machine Learning')

model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/books_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        poster_url.append(url)

    return poster_url


def recommend_book(book_name, num_recommendations=10):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=num_recommendations)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url


selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

search_button = st.button('Recommend')

if search_button:
    recommended_books, poster_url = recommend_book(selected_books, num_recommendations=10)

    # Create a 5-column grid layout for recommendations
    cols = st.columns(5)

    for i in range(len(recommended_books)):
        with cols[i % 5]:
            # Adjust the size and style of book recommendations
            st.text(recommended_books[i])
            st.image(poster_url[i], width=150)  # Adjust width as needed

    # Add an optional horizontal line to separate the recommendations
    st.markdown("---")

# Add additional styling or elements as needed to make the app more attractive
