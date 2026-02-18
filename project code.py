import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def recommend_movie(title, n=5):
    idx = movies[movies["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

st.title("🎬 Movie Recommendation Website")

selected_movie = st.selectbox(
    "Search or select a movie",
    movies["title"].values
)

if st.button("Recommend"):
    recommendations = recommend_movie(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
