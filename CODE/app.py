from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import pandas as pd # type: ignore
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  #type: ignore

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Preprocessed Data ---
with open('movies_with_ratings.pkl', 'rb') as f:
    movies_with_ratings = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocess and vectorize movie features using cosine similarity
def vectorize_movies():
    """
    Vectorize movie descriptions or genres into numerical embeddings using TF-IDF
    for similarity computations.
    """
    # Process genres or use preprocessed text
    movie_texts = movies_with_ratings['genres'].fillna("").str.lower()
    # Vectorize the processed text using TFIDF
    movie_vectors = vectorizer.transform(movie_texts)
    return movie_vectors


# Recommendation logic using cosine similarity
def find_similar_movies(query, top_k=20):
    """
    Compute cosine similarity between a given query and movies' descriptions,
    then return the top-k similar movies.
    """
    # Vectorize all movies' features
    movie_vectors = vectorize_movies()
    
    # Transform the query into a vector
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, movie_vectors).flatten()
    
    # Add similarity scores to the movie DataFrame
    movies_with_ratings['similarity'] = similarities
    
    # Sort by similarity and fetch top K movies
    recommended_movies = movies_with_ratings.sort_values(
    by=['similarity', 'average_rating'], 
    ascending=[False, False]
).head(top_k)

    
    # Return only relevant columns for response
    return recommended_movies[['title', 'genres', 'average_rating', 'release_year', 'similarity']]


# --- Flask Routes ---
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the recommendation requests.
    The user sends a keyword as input, and similar movies are computed.
    """
    query = request.form.get('query')
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        # Find similar movies based on the given query
        recommendations = find_similar_movies(query)
        
        # Format response data
        records = recommendations.to_dict(orient='records')
        #print(records)
        return render_template('movies.html',records=records)
    except Exception as e:
        return jsonify({'error': f"An internal error occurred: {str(e)}"}), 500


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
