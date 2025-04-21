# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:55:25 2025

@author: Ross E & Miona M
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load movie data
movies = pd.read_csv('/Users/Ross E/Downloads/Movielens Dataset/movie.csv').head(1000)

# 🔹 Create fake user history: randomly pick 100 movies the user has "watched"
def generate_user_history(user_id, num_movies=100):
    watched_movie_ids = random.sample(list(movies['movieId']), num_movies)
    return watched_movie_ids

# 🔹 Combine genres into one 'tags' column for similarity checking
movies['tags'] = movies['genres'].str.replace('|', ' ')

# 🔹 Convert 'tags' to numerical vectors using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf.fit_transform(movies['tags'])

# 🔹 Calculate cosine similarity between all movies based on tags
cosine_sim = cosine_similarity(tag_matrix)

# 🔹 Function to recommend movies based on genre similarity to watched ones
def recommend_movies_based_on_genre(watched_movie_ids, num_recommendations=5):
    sim_scores = []

    for movie_id in watched_movie_ids:
        if movie_id in movies['movieId'].values:
            idx = movies[movies['movieId'] == movie_id].index[0]
            sim = list(enumerate(cosine_sim[idx]))

            # Add scores to the list
            sim_scores.extend(sim)

    # Sort by similarity score and remove duplicates
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    seen_indices = [movies[movies['movieId'] == mid].index[0] for mid in watched_movie_ids if mid in movies['movieId'].values]
    recommended = []
    for i, score in sim_scores:
        if i not in seen_indices and movies.iloc[i]['movieId'] not in watched_movie_ids:
            recommended.append(movies.iloc[i]['title'])
        if len(recommended) >= num_recommendations:
            break

    return recommended

# 🔹 Get user input for User ID
print("\n***Welcome To Movielens***")
user_id = int(input("\nEnter your USER ID (e.g., 1, 2, 3, etc.): "))

# 🔹 Generate a "fake" user history based on the input User ID
watched_movie_ids = generate_user_history(user_id)

# 🔹 Get movie recommendations based on the user's history
recommendations = recommend_movies_based_on_genre(watched_movie_ids, 5)

# 🔹 Print recommendations in a nice format
print(f"\n🎬 *** Recommended Movies for User {user_id} ***\n")
for i, title in enumerate(recommendations, start=1):
    print(f"{i}. {title}")

