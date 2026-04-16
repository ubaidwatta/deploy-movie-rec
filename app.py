import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Genre Based Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #aaaaaa;
        margin-bottom: 30px;
    }
    .movie-card {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data_and_build_model():
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        st.error("Missing dataset files!")
        st.stop()

    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    similarity_matrix = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(similarity_matrix,
                                 index=user_movie_matrix.index,
                                 columns=user_movie_matrix.index)

    original_user_ids = user_movie_matrix.index.tolist()

    return movies, ratings, similarity_df, original_user_ids


movies, ratings, similarity_df, original_user_ids = load_data_and_build_model()

# --- Extract Unique Genres ---
all_genres = set()
for g in movies['genres']:
    for genre in g.split('|'):
        all_genres.add(genre)

all_genres = sorted(list(all_genres))

# --- Recommendation Function ---
def get_recommendations(custom_user_id, selected_genres=None, num_recommendations=5):
    try:
        original_user_id = original_user_ids[custom_user_id - 1]
    except IndexError:
        return pd.DataFrame()

    if selected_genres:
        pattern = '|'.join(selected_genres)
        filtered_movies = movies[movies['genres'].str.contains(pattern, case=False, na=False)]
    else:
        filtered_movies = movies

    similar_users = similarity_df[original_user_id].sort_values(ascending=False)[1:num_recommendations + 1].index

    recommended_movies = ratings[ratings['userId'].isin(similar_users)]

    movie_recommendations = (
        recommended_movies[recommended_movies['movieId'].isin(filtered_movies['movieId'])]
        .groupby('movieId')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(num_recommendations)
    )

    recommended_movies_details = pd.merge(
        movie_recommendations.reset_index(), movies, on='movieId'
    )

    return recommended_movies_details[['title', 'genres', 'rating']]


# --- Header ---
st.markdown('<div class="title">🎬 Genre Based Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies based on users with similar taste</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🔧 Filters")

custom_user_id = st.sidebar.number_input(
    "User ID",
    min_value=1,
    max_value=len(original_user_ids),
    value=1
)

selected_genres = st.sidebar.multiselect(
    "Select Genres (optional)",
    all_genres
)

num_recommendations = st.sidebar.slider("Number of recommendations", 3, 15, 5)

# --- Button ---
if st.sidebar.button("🎯 Get Recommendations"):
    with st.spinner("Finding the best movies for you..."):
        recommended_movies = get_recommendations(
            custom_user_id,
            selected_genres,
            num_recommendations
        )

        if not recommended_movies.empty:
            st.success("✅ Recommendations ready!")

            if selected_genres:
                st.info(f"Filtering by: {', '.join(selected_genres)}")

            cols = st.columns(2)

            for i, row in recommended_movies.iterrows():
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <h4>🎬 {row['title']}</h4>
                            <p><b>Genre:</b> {row['genres']}</p>
                            <p><b>⭐ Rating:</b> {round(row['rating'], 2)}</p>
                        </div>
                    """, unsafe_allow_html=True)

        else:
            st.warning("❌ No recommendations found. Try another input.")