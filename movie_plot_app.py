

import streamlit as st
import wikipediaapi
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import joblib

# Define your user agent for Wikipedia API
user_agent = "StreamlitMoviePlotApp/1.0 (your-email@example.com)"

# Load the trained model and other components
model = load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.joblib', 'rb') as le_handle:
    label_encoder = joblib.load(le_handle)

# Load your DataFrame with movie titles and genres
df = pd.read_csv("no_null_vals2.csv")

# Set your max sequence length
max_length = 175  # This should match the value used during training

# Setup Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent=user_agent
)

def get_movie_plot(movie_name):
    page = wiki_wiki.page(movie_name)
    if not page.exists():
        return None, "Page on Wikipedia does not exist for this movie."
    plot_section = page.section_by_title('Plot')
    if plot_section:
        return plot_section.text, None
    else:
        return None, "Plot section not found for this movie."

def predict_genre(plot):
    # Preprocess the plot text
    sequence = tokenizer.texts_to_sequences([plot])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Predict the genre
    prediction_probs = model.predict(padded_sequence)
    predicted_genre_index = np.argmax(prediction_probs)
    
    # Make sure to convert the index to a list before calling inverse_transform
    predicted_genre = label_encoder.inverse_transform([predicted_genre_index])
    
    # Obtain the highest confidence value
    confidence = np.max(prediction_probs)
    
    return predicted_genre[0], confidence

def get_recommendations(predicted_genre):
    # Filter the DataFrame based on the predicted genre
    recommended_movies = df[df['Genre Names'].astype(str).str.contains(predicted_genre, case=False, na=False)]
    
    # Get up to 5 recommended movie titles
    return recommended_movies['title'].tolist()[:5]

st.title('Movie Genre Predictor')

# User input for movie name
movie_name = st.text_input("Enter the movie name:")

if st.button('Predict Genre'):
    if movie_name:
        # Fetch the plot from Wikipedia
        plot, error = get_movie_plot(movie_name)
        
        if error:
            st.error(error)
        else:
            # Predict and display the genre
            predicted_genre, confidence = predict_genre(plot)
            st.success(f"Predicted Genre: {predicted_genre} (Confidence: {confidence:.2f})")
            
            # Get and display recommended movies from the same genre
            recommendations = get_recommendations(predicted_genre)
            if recommendations:
                st.write("Movies you might like:")
                for title in recommendations:
                    st.write(title)
            else:
                st.write("No movie recommendations found.")
    else:
        st.error("Please enter a movie name.")


