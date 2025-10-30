import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib # <-- NEW: For loading models
import tempfile

# -----------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# (This section is now MUCH smaller)
# -----------------------------------------------------------------

# --- Setup Tesseract & Poppler Paths ---
# (We still need these for processing NEWLY UPLOADED files)
try:
    tesseract_install_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tesseract_install_path
except Exception:
    pass 
poppler_install_path = r"C:\poppler\bin"

# --- NLTK Setup ---
@st.cache_data
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
setup_nltk()

# --- PDF Extraction & Preprocessing ---
# (These are the same as before, needed for new uploads)
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc: text += page.get_text()
        if text.strip(): return text
    except Exception: pass
    try:
        # We don't need Tika anymore for the main db,
        # so we can remove it to make the app lighter.
        # But let's keep the OCR fallback.
        images = convert_from_path(pdf_path, poppler_path=poppler_install_path)
        ocr_text = ""
        for img in images: ocr_text += pytesseract.image_to_string(img, lang='eng')
        if ocr_text.strip(): return ocr_text
    except Exception: return ""
    return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# -----------------------------------------------------------------
# 2. MODEL LOADING & PREPARATION (*** COMPLETELY NEW ***)
# -----------------------------------------------------------------

@st.cache_resource
def load_models_and_data():
    print("Loading pre-processed files from disk...")
    
    # --- 1. Load Author Database ---
    author_df = pd.read_parquet('author_database.parquet')
    
    # --- 2. Load TF-IDF Model ---
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    tfidf_matrix = joblib.load('tfidf_matrix.joblib')
    
    # --- 3. Load Embedding Model ---
    model = SentenceTransformer('all-MiniLM-L6-v2')
    paper_embeddings = np.load('paper_embeddings.npy')
    
    # --- 4. Build Reviewer-Reviewer Similarity Model ---
    author_vectors_df = (
        pd.DataFrame(paper_embeddings)
        .set_index(author_df['author'])
        .groupby(level=0)
        .mean()
    )
    author_similarity_matrix = cosine_similarity(author_vectors_df)
    author_sim_df = pd.DataFrame(
        author_similarity_matrix,
        index=author_vectors_df.index,
        columns=author_vectors_df.index
    )
    
    print("All models loaded successfully!")
    return author_df, tfidf_vectorizer, tfidf_matrix, model, paper_embeddings, author_sim_df

# --- Load all data and models (This will be very fast now) ---
st.write("Loading all models and preparing data...")
author_df, tfidf_vectorizer, tfidf_matrix, model, paper_embeddings, author_sim_df = load_models_and_data()
st.success("All models and data loaded successfully! ✅")

# -----------------------------------------------------------------
# 3. RECOMMENDATION FUNCTIONS (Unchanged)
# -----------------------------------------------------------------

def recommend_with_tfidf(processed_input, k=5):
    input_vector = tfidf_vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    author_df['similarity_score'] = similarities
    top_authors = author_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
    return top_authors.head(k)

def recommend_with_embeddings(processed_input, k=5):
    input_embedding = model.encode([processed_input])
    similarities = cosine_similarity(input_embedding, paper_embeddings).flatten()
    author_df['similarity_score'] = similarities
    top_authors = author_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
    return top_authors.head(k)

def get_similar_reviewers(author_name, k=5):
    if author_name not in author_sim_df:
        return pd.Series(dtype='float64')
    similar_scores = author_sim_df[author_name]
    top_similar = similar_scores.sort_values(ascending=False).drop(author_name)
    return top_similar.head(k)

# -----------------------------------------------------------------
# 4. UI: PAPER RECOMMENDATION (Unchanged)
# -----------------------------------------------------------------

st.title("👨‍🎓 Research Paper Reviewer Recommendation System")
st.write("This tool uses NLP to find the best expert reviewers for your paper.")

# (All the session state logic is unchanged)
if 'k_value' not in st.session_state: st.session_state.k_value = 5
if 'file_bytes' not in st.session_state: st.session_state.file_bytes = None
if 'file_name' not in st.session_state: st.session_state.file_name = ""
if 'results' not in st.session_state: st.session_state.results = None

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
st.number_input("Number of reviewers to recommend (k):", min_value=3, max_value=20, key='k_value')
run_button = st.button("Find Reviewers for Paper")

if uploaded_file is not None:
    st.session_state.file_bytes = uploaded_file.getvalue()
    st.session_state.file_name = uploaded_file.name
    st.session_state.results = None
    st.info(f"Loaded new file: **{st.session_state.file_name}**")

if run_button:
    if st.session_state.file_bytes is not None:
        with st.spinner("Processing your paper..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(st.session_state.file_bytes)
                tmp_file_path = tmp_file.name

            input_text = extract_text_from_pdf(tmp_file_path)
            processed_input = preprocess_text(input_text)
            k = st.session_state.k_value
            
            # --- Get Raw Recommendations ---
            tfidf_recs_raw = recommend_with_tfidf(processed_input, k=k+1) # Get k+1 initially
            embedding_recs_raw = recommend_with_embeddings(processed_input, k=k+1) # Get k+1 initially
            
            os.remove(tmp_file_path)

            # --- *** NEW: FILTERING LOGIC *** ---
            # Check if the uploaded file name matches any paper in our database
            # We use the original filename stored in session state
            uploaded_filename = st.session_state.file_name
            
            # Find if this paper exists in our original dataframe
            matching_paper = author_df[author_df['paper'] == uploaded_filename]
            
            author_to_exclude = None
            if not matching_paper.empty:
                # If it matches, get the author of that paper from our database
                author_to_exclude = matching_paper.iloc[0]['author']
                print(f"Uploaded paper matches dataset paper. Excluding author: {author_to_exclude}")

            # Filter the recommendations if an author needs exclusion
            if author_to_exclude:
                tfidf_recs_filtered = tfidf_recs_raw[tfidf_recs_raw.index != author_to_exclude].head(k)
                embedding_recs_filtered = embedding_recs_raw[embedding_recs_raw.index != author_to_exclude].head(k)
            else:
                # No match found, or no author to exclude, just take top k
                tfidf_recs_filtered = tfidf_recs_raw.head(k)
                embedding_recs_filtered = embedding_recs_raw.head(k)
            # --- *** END OF FILTERING LOGIC *** ---

            # Store the FILTERED results in session state
            st.session_state.results = (tfidf_recs_filtered, embedding_recs_filtered)
    
    elif st.session_state.file_bytes is None:
        st.error("Please upload a PDF file first.")

if st.session_state.results is not None:
    st.success("Analysis complete!")
    st.subheader(f"Top {st.session_state.k_value} Recommended Reviewers for: **{st.session_state.file_name}**")
    tfidf_recs, embedding_recs = st.session_state.results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Method 1: TF-IDF (Keyword Match)")
        st.dataframe(pd.DataFrame({'Author': tfidf_recs.index, 'Score': tfidf_recs.values}))
    with col2:
        st.markdown("#### Method 2: Embeddings (Semantic Match)")
        st.dataframe(pd.DataFrame({'Author': embedding_recs.index, 'Score': embedding_recs.values}))
    with st.expander("ℹ️ How to interpret these results:"):
        st.markdown("""
        * **TF-IDF (Keyword Match):** Finds reviewers who used the *exact same keywords*.
        * **Embeddings (Semantic Match):** Finds reviewers who wrote about the *same concepts*, even with different words.
        """)

# -----------------------------------------------------------------
# 5. UI: REVIEWER-REVIEWER SIMILARITY (Unchanged)
# -----------------------------------------------------------------

st.markdown("---")
st.header("🤝 Find Similar Reviewers ")
st.write("Select an author from the database to find other authors with similar expertise.")

all_authors = sorted(author_df['author'].unique())
selected_author = st.selectbox("Select an author:", all_authors)

if st.button(f"Find Reviewers Similar to {selected_author}"):
    with st.spinner(f"Analyzing authors similar to {selected_author}..."):
        similar_authors = get_similar_reviewers(selected_author, k=5)
        st.subheader(f"Top 5 Reviewers with Similar Expertise:")
        st.dataframe(pd.DataFrame({
            'Author': similar_authors.index,
            'Similarity Score': similar_authors.values

        }))
