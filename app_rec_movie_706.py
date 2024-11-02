
import streamlit as st
from surprise import SVD
import pandas as pd
import pickle

# Load data back from the file
with open('recommendation_movie_svd_706.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Title for the app
st.title("Movie Recommendations")

# User input for user ID
user_id = st.number_input("Enter User ID", min_value=1, step=1)

# Get rated and unrated movies for the given user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

# Make predictions on unrated movies
pred_ratings = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

# Sort predictions by estimated rating in descending order
sorted_predictions = sorted(pred_ratings, key=lambda x: x.est, reverse=True)

# Get top 10 movie recommendations
top_recommendations = sorted_predictions[:10]

# Display recommendations
st.write(f"### Top 10 movie recommendations for User {user_id}:")
for recommendation in top_recommendations:
    movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
    st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
