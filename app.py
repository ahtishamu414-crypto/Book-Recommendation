import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Load pickle files
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("book_pivot.pkl", "rb") as f:
    book_pivot = pickle.load(f)

with open("books.pkl", "rb") as f:
    books = pickle.load(f)

# -------------------------------
# Add poster fetching function
# -------------------------------
def get_poster(isbn):
    """Return Open Library cover URL from ISBN"""
    if pd.isna(isbn) or isbn == "":
        return None
    return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"

# Add image_url column if missing
if "image_url" not in books.columns and "isbn" in books.columns:
    books["image_url"] = books["isbn"].apply(get_poster)

# -------------------------------
# Recommend Function
# -------------------------------
def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1),
            n_neighbors=5  # 1 input + 4 similar
        )

        recommended_books = []
        for i in suggestions[0]:
            if book_pivot.index[i] != book_name:
                title = book_pivot.index[i]

                details = books[books["title"] == title].drop_duplicates("title")

                if not details.empty:
                    author = details["author"].values[0]
                    image = details["image_url"].values[0] if "image_url" in details.columns else None
                else:
                    author = "Unknown"
                    image = None

                recommended_books.append({
                    "title": title,
                    "author": author,
                    "image": image
                })

        return recommended_books[:4]  # only 4 recommendations

    except IndexError:
        return []

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Book Recommendation System")
st.write("Select a book and get **4 similar recommendations** with covers")

selected_book = st.selectbox(
    "Select a Book:",
    book_pivot.index.tolist()
)

if st.button("Recommend"):
    st.subheader(f"Books similar to: **{selected_book}**")
    recommendations = recommend_book(selected_book)

    if recommendations:
        cols = st.columns(4)  # all 4 in a single row
        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                if rec["image"]:
                    st.image(rec["image"], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/150x220?text=No+Image")
                st.markdown(f"**{rec['title']}**")
                st.caption(f"👤 {rec['author']}")
    else:
        st.error("No recommendations found!")
