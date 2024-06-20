# Import libraries
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Environment variables
from dotenv import load_dotenv
import os
load_dotenv()
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
# Authentication using SpotifyClientCredentials
auth = SpotifyClientCredentials(client_id, client_secret)
spotify = spotipy.Spotify(auth = auth, requests_timeout = 10, retries = 10)

# Helper function to get track features
def get_track_features(trackID):
    try:
        # Authentication key
        spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        # Using track() and audio_features() calls on Spotify API
        # 1. We are going to get the track info and audio features of a sample song and turn it into a dataframe
        trackInfo = spotify.track(trackID) 
        trackName = trackInfo['name']
        trackID = trackInfo['id']
        trackArtists = [artist['name'] for artist in trackInfo['artists']]
        trackAlbum = trackInfo['album']['name']
        trackDuration = trackInfo['duration_ms']
        trackPopularity = trackInfo['popularity']
        # Getting audio features
        audioFeatures = spotify.audio_features(trackID)
        if audioFeatures:
            audioFeatures = audioFeatures[0]
            # Get genre for first artist on the track
            artistID = trackInfo['artists'][0]['id']
            artistInfo = spotify.artist(artistID)
            artistGenres = artistInfo.get('genres', []) # Create an empty list to store artist genres
            # Add track and audio features to the list
            trackFeatures = ({
                'Track Name': trackName,
                'Track ID': trackID,
                'Artists': ', '.join(trackArtists),
                'Album': trackAlbum,
                'Genre': ', '.join(artistGenres),
                'Duration (ms)': trackDuration,
                'Popularity': trackPopularity,
                'Danceability': audioFeatures['danceability'],
                'Energy': audioFeatures['energy'],
                'Acousticness': audioFeatures['acousticness'],
                'Instrumentalness': audioFeatures['instrumentalness'],
                'Liveness': audioFeatures['liveness'],
                'Loudness': audioFeatures['loudness'],
                'Speechiness': audioFeatures['speechiness'],
                'Key': audioFeatures['key'],
                'Tempo': audioFeatures['tempo']
            })
        # Making the final track variable for our data as a dataframe
        track_data = pd.DataFrame([trackFeatures])
        return(track_data)
    except Exception as e:
        print(f'Error fetching track features for trackID {trackID}: {e}')
        return None


def get_recommendations(song_id, songs_displayed=10):

    # More data preprocessing
    sample_data = pd.read_csv('spotifysampledata.csv')
    # Drop rows with empty values in the 'Artists' column, replace NaN genre with empty string, drop duplicates
    sample_data = sample_data.dropna(subset=['Artists'])
    sample_data['Genre'] = sample_data['Genre'].fillna('')
    sample_data = sample_data.drop_duplicates(subset='Track ID', keep='first')
    sample_data.reset_index(drop=True, inplace=True)

    # Add song to matrix if not already in it
    if song_id not in sample_data['Track ID'].values:
        song_df = get_track_features(song_id)
        sample_data = pd.concat([sample_data, song_df], ignore_index = True) # Concatenate to existing dataframe

    # TF-IDF vectorization (for text):
    tfidf_vectorizer_artists = TfidfVectorizer(stop_words='english')
    tfidf_artists_matrix = tfidf_vectorizer_artists.fit_transform(sample_data['Artists'])
    tfidf_vectorizer_genres = TfidfVectorizer(stop_words='english')
    tfidf_genres_matrix = tfidf_vectorizer_genres.fit_transform(sample_data['Genre'])
    # print(type(tfidf_artists_matrix))

    # Scale/normalize numerical features (make sure they are of same importance)
    scaler = StandardScaler()
    numerical_features = ['Popularity', 'Danceability', 'Energy', 'Acousticness', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Tempo']
    numerical_features_scaled = scaler.fit_transform(sample_data[numerical_features])
    numerical_features_sparse = csr_matrix(numerical_features_scaled)
    # print(type(numerical_features_scaled))

    # Combine TF-IDF vectorization and scaled numerical features (horizontally stack matrix and arrays)
    combined_features = hstack([tfidf_artists_matrix, tfidf_genres_matrix, numerical_features_sparse])
    # print(combined_features.shape)

    # Get cosine similarity of the X and Y of our combined features
    cosim_matrix = cosine_similarity(combined_features, combined_features)
    # Convert to dataframe
    cosim_df = pd.DataFrame(cosim_matrix, index = sample_data['Track ID'], columns = sample_data['Track ID'])
    # Get similarity score of chosen song + most similar songs in the dataset
    sim_scores = cosim_df[song_id]
    sim_scores = sim_scores.sort_values(ascending=False)
    recommended_songs_index = sim_scores.index[1:(songs_displayed+1)]
    recommended_songs = sample_data[sample_data['Track ID'].isin(recommended_songs_index)][['Track Name', 'Album', 'Artists']].reset_index(drop=True)
    recommended_songs.index = recommended_songs.index + 1

    return recommended_songs

recommendations = get_recommendations('3gG6t4xCPtnTskhpRFFsqO', 5)
print(recommendations)