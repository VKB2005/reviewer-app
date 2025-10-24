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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # Added CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation # Added LDA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import tempfile

# -----------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# -----------------------------------------------------------------

# --- Setup Paths ---
# Tesseract path (Needed if not in system PATH, comment out if deployed)
# try:
#     tesseract_install_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#     pytesseract.pytesseract.tesseract_cmd = tesseract_install_path
# except Exception:
#     print("Tesseract command not set. Ensure Tesseract is in your PATH.")
#     pass

# Use the simplified Poppler path
poppler_install_path = r"C:\Users\VAMSHI KRISHNA BABU\poppler\poppler-25.07.0\Library\bin"

# --- NLTK Setup ---
# --- NLTK Setup ---
@st.cache_data
def setup_nltk():
    try:
        # Check if resources exist, raise LookupError if not
        nltk.data.find('corpora/stopwords')
        print("NLTK stopwords found.")
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt tokenizer found.")
    except LookupError:
        print("NLTK punkt tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
    try:
        # This resource might be needed by word_tokenize indirectly
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK punkt_tab resource found.")
    except LookupError:
        print("NLTK punkt_tab resource not found. Downloading...")
        nltk.download('punkt_tab', quiet=True) # Added this download

setup_nltk()

# --- PDF Extraction & Preprocessing (No Tika) ---
def extract_text_from_pdf(pdf_path):
    try: # PyMuPDF
        text = ""
        with fitz.open(pdf_path) as doc:
            # Check for encryption first
            if doc.is_encrypted:
                print(f"WARN: {os.path.basename(pdf_path)} is encrypted. Skipping.")
                return ""
            for page in doc: text += page.get_text()
        if text.strip(): return text
    except Exception as e_fitz:
        print(f"INFO: PyMuPDF failed on {os.path.basename(pdf_path)} ({e_fitz}). Trying OCR.")
        pass # Fall through to OCR
    try: # OCR Fallback
        images = convert_from_path(pdf_path, poppler_path=poppler_install_path)
        if not images: # Handle case where pdf2image returns empty list
             print(f"WARN: pdf2image couldn't convert {os.path.basename(pdf_path)}. Skipping OCR.")
             return ""
        ocr_text = ""
        for img in images:
            try:
                ocr_text += pytesseract.image_to_string(img, lang='eng')
            except pytesseract.TesseractNotFoundError:
                 st.error("Tesseract is not installed or not in your PATH. OCR functionality will not work.")
                 print("ERROR: Tesseract not found during OCR.")
                 return "" # Stop processing if Tesseract isn't found
            except Exception as e_tess:
                 print(f"WARN: Tesseract failed on an image from {os.path.basename(pdf_path)} ({e_tess})")
                 continue # Try next image
        if ocr_text.strip():
             print(f"INFO: Successfully extracted text using OCR for {os.path.basename(pdf_path)}")
             return ocr_text
    except Exception as e_ocr:
        # Catch errors during pdf2image conversion itself
        print(f"ERROR: OCR stage failed for {os.path.basename(pdf_path)} ({e_ocr}). Might be Poppler path issue or corrupted PDF.")
        return ""
    print(f"WARN: Both PyMuPDF & OCR failed for {os.path.basename(pdf_path)}. Skipping.")
    return ""


def preprocess_text(text):
    if not isinstance(text, str): # Add check for non-string input
         return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Handle case where text becomes empty after preprocessing
    return " ".join(filtered_tokens) if filtered_tokens else ""


# -----------------------------------------------------------------
# 2. MODEL LOADING & PREPARATION (Includes LDA)
# -----------------------------------------------------------------

@st.cache_resource
def load_models_and_data():
    print("Loading pre-processed files...")
    # Load base data
    try:
        author_df = pd.read_parquet('author_database.parquet')
        paper_embeddings = np.load('paper_embeddings.npy')
    except FileNotFoundError as e:
        st.error(f"Error loading base data ({e}). Make sure '.parquet' and '.npy' files are in the app directory.")
        print(f"FATAL ERROR: Could not load base data files: {e}")
        st.stop() # Stop the app if base files are missing

    # Load TF-IDF
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
        tfidf_matrix = joblib.load('tfidf_matrix.joblib')
    except FileNotFoundError as e:
        st.error(f"Error loading TF-IDF models ({e}). Make sure '.joblib' files are present.")
        print(f"FATAL ERROR: Could not load TF-IDF model files: {e}")
        st.stop()

    # Load Sentence Transformer (from local cache)
    try:
        # Check if cache folder exists before loading
        if os.path.isdir('./model_cache'):
             model = SentenceTransformer('./model_cache')
        else:
             # Fallback to downloading if cache doesn't exist (useful for first run)
             print("Local model cache not found. Downloading SentenceTransformer model...")
             model = SentenceTransformer('all-MiniLM-L6-v2')
             model.save('./model_cache') # Save it for next time
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        print(f"FATAL ERROR: Could not load Sentence Transformer model: {e}")
        st.stop()


    # Load LDA components
    try:
        lda_count_vectorizer = joblib.load('lda_count_vectorizer.joblib')
        lda_model = joblib.load('lda_model.joblib')
        lda_author_profiles = joblib.load('lda_author_profiles.joblib')
    except FileNotFoundError as e:
        st.error(f"Error loading LDA models ({e}). Make sure LDA '.joblib' files are present.")
        print(f"FATAL ERROR: Could not load LDA model files: {e}")
        st.stop()


    # Build Reviewer-Reviewer Similarity Model
    print("Building reviewer similarity matrix...")
    valid_indices = [i for i, emb in enumerate(paper_embeddings) if author_df.iloc[i]['author'] is not None] # Ensure author exists
    author_vectors_df = (
        pd.DataFrame(paper_embeddings[valid_indices])
        .set_index(author_df.iloc[valid_indices]['author'])
        .groupby(level=0)
        .mean()
    )
    # Filter out any potential rows with NaN embeddings if preprocessing failed unexpectedly
    author_vectors_df = author_vectors_df.dropna()

    if author_vectors_df.empty:
         st.error("Could not build author similarity profiles. Check data quality.")
         print("FATAL ERROR: Author vectors DataFrame is empty after processing.")
         author_sim_df = pd.DataFrame() # Create empty DF to avoid crashing later
    else:
        author_similarity_matrix = cosine_similarity(author_vectors_df)
        author_sim_df = pd.DataFrame(
            author_similarity_matrix,
            index=author_vectors_df.index,
            columns=author_vectors_df.index
        )

    print("All models loaded!")
    return (author_df, paper_embeddings,
            tfidf_vectorizer, tfidf_matrix,
            model,
            lda_count_vectorizer, lda_model, lda_author_profiles,
            author_sim_df)

# --- Load all models ---
st.write("Loading all models...")
(author_df, paper_embeddings,
 tfidf_vectorizer, tfidf_matrix,
 model,
 lda_count_vectorizer, lda_model, lda_author_profiles,
 author_sim_df) = load_models_and_data()
st.success("All models loaded! ✅")

# -----------------------------------------------------------------
# 3. RECOMMENDATION FUNCTIONS (Includes LDA)
# -----------------------------------------------------------------

def recommend_with_tfidf(processed_input, k=5):
    if not processed_input: return pd.Series(dtype='float64') # Handle empty input
    input_vector = tfidf_vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    # Create a temporary df to avoid modifying global state if running concurrently
    temp_df = pd.DataFrame({'author': author_df['author'], 'similarity_score': similarities})
    top_authors = temp_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
    return top_authors.head(k)

def recommend_with_embeddings(processed_input, k=5):
    if not processed_input: return pd.Series(dtype='float64') # Handle empty input
    input_embedding = model.encode([processed_input])
    # Ensure paper_embeddings is 2D
    embeddings_matrix = paper_embeddings if paper_embeddings.ndim == 2 else paper_embeddings.reshape(-1, 1) # Reshape if needed, adjust dimensions as per your model output
    if input_embedding.shape[1] != embeddings_matrix.shape[1]:
         print(f"ERROR: Embedding dimension mismatch. Input: {input_embedding.shape}, Database: {embeddings_matrix.shape}")
         return pd.Series(dtype='float64')
    similarities = cosine_similarity(input_embedding, embeddings_matrix).flatten()
    temp_df = pd.DataFrame({'author': author_df['author'], 'similarity_score': similarities})
    top_authors = temp_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
    return top_authors.head(k)

def recommend_with_lda(processed_input, k=5):
    if not processed_input: return pd.Series(dtype='float64') # Handle empty input
    try:
        input_count_vector = lda_count_vectorizer.transform([processed_input])
        input_topic_distribution = lda_model.transform(input_count_vector)
        # Ensure author profiles are valid before calculating similarity
        if lda_author_profiles.empty:
             print("ERROR: LDA Author profiles are empty.")
             return pd.Series(dtype='float64')
        similarities = cosine_similarity(input_topic_distribution, lda_author_profiles.values).flatten()
        author_scores = pd.Series(similarities, index=lda_author_profiles.index)
        top_authors = author_scores.sort_values(ascending=False)
        return top_authors.head(k)
    except Exception as e:
        print(f"Error during LDA recommendation: {e}")
        return pd.Series(dtype='float64')


def get_similar_reviewers(author_name, k=5):
    # Check if author_sim_df is valid
    if author_sim_df.empty or author_name not in author_sim_df:
        print(f"Author '{author_name}' not found in similarity matrix or matrix is empty.")
        return pd.Series(dtype='float64')
    similar_scores = author_sim_df[author_name]
    top_similar = similar_scores.sort_values(ascending=False).drop(author_name)
    return top_similar.head(k)

# -----------------------------------------------------------------
# 4. UI: PAPER RECOMMENDATION (Includes LDA and Filtering)
# -----------------------------------------------------------------

st.title("👨‍🎓 Research Paper Reviewer Recommendation System")
st.write("Upload a PDF to find the best expert reviewers.")

# Session state
if 'k_value' not in st.session_state: st.session_state.k_value = 5
if 'file_bytes' not in st.session_state: st.session_state.file_bytes = None
if 'file_name' not in st.session_state: st.session_state.file_name = ""
if 'results' not in st.session_state: st.session_state.results = None # Holds 3 results

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
st.number_input("Number of reviewers to recommend (k):", min_value=3, max_value=20, key='k_value')
run_button = st.button("Find Reviewers for Paper")

if uploaded_file is not None:
    # If a new file is uploaded, store it and clear old results
    if uploaded_file.name != st.session_state.get('last_uploaded_filename', ''):
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name
        st.session_state.results = None
        st.session_state.last_uploaded_filename = uploaded_file.name # Track the last file
        st.info(f"Loaded: **{st.session_state.file_name}**")
    else:
        # Keep existing file if the uploader widget re-runs without a new file selection
        pass


if run_button:
    if st.session_state.file_bytes is not None:
        with st.spinner("Processing paper and finding reviewers..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(st.session_state.file_bytes)
                tmp_file_path = tmp_file.name

            input_text = extract_text_from_pdf(tmp_file_path)

            if not input_text:
                 st.error(f"Could not extract text from the uploaded PDF: {st.session_state.file_name}. It might be empty, encrypted, corrupted, or purely image-based with OCR failing. Please try another file.")
                 st.session_state.results = None
                 if os.path.exists(tmp_file_path): os.remove(tmp_file_path)

            else:
                processed_input = preprocess_text(input_text)

                # Check if processed input is empty after cleaning
                if not processed_input:
                     st.warning(f"Could not extract meaningful text content (after cleaning) from: {st.session_state.file_name}. Recommendations might be inaccurate.")
                     # Set empty results if no text
                     st.session_state.results = (pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64'))
                else:
                    k = st.session_state.k_value
                    # Get Raw Recommendations (k+1 for filtering)
                    tfidf_recs_raw = recommend_with_tfidf(processed_input, k=k+5) # Get more for robust filtering
                    embedding_recs_raw = recommend_with_embeddings(processed_input, k=k+5)
                    lda_recs_raw = recommend_with_lda(processed_input, k=k+5)

                    # Filter out original author
                    uploaded_filename = st.session_state.file_name
                    matching_paper = author_df[author_df['paper'] == uploaded_filename]
                    author_to_exclude = None
                    if not matching_paper.empty:
                        author_to_exclude = matching_paper.iloc[0]['author']

                    if author_to_exclude:
                        print(f"Excluding author: {author_to_exclude}")
                        tfidf_recs = tfidf_recs_raw[tfidf_recs_raw.index != author_to_exclude].head(k)
                        embedding_recs = embedding_recs_raw[embedding_recs_raw.index != author_to_exclude].head(k)
                        lda_recs = lda_recs_raw[lda_recs_raw.index != author_to_exclude].head(k)
                    else:
                        tfidf_recs = tfidf_recs_raw.head(k)
                        embedding_recs = embedding_recs_raw.head(k)
                        lda_recs = lda_recs_raw.head(k)

                    # Store results
                    st.session_state.results = (tfidf_recs, embedding_recs, lda_recs)

                # Cleanup temp file regardless of outcome
                if os.path.exists(tmp_file_path): os.remove(tmp_file_path)

    elif st.session_state.file_bytes is None:
        st.error("Please upload a PDF file first.")


# Display results if they exist
if st.session_state.results is not None:
    tfidf_recs, embedding_recs, lda_recs = st.session_state.results

    # Check if any recommendations were actually generated
    if tfidf_recs.empty and embedding_recs.empty and lda_recs.empty and st.session_state.file_name:
         # This happens if processed_input was empty or models failed
         st.warning(f"No recommendations could be generated for {st.session_state.file_name}. This might happen if the PDF content was too short or unprocessable after cleaning.")
    elif st.session_state.file_name: # Only show results section if a file was processed
        st.success("Analysis complete!")
        st.subheader(f"Top {st.session_state.k_value} Recommended Reviewers for: **{st.session_state.file_name}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### TF-IDF (Keywords)")
            st.dataframe(pd.DataFrame({'Author': tfidf_recs.index, 'Score': tfidf_recs.values}))
        with col2:
            st.markdown("##### Embeddings (Semantic)")
            st.dataframe(pd.DataFrame({'Author': embedding_recs.index, 'Score': embedding_recs.values}))
        with col3:
            st.markdown("##### LDA (Topics)")
            st.dataframe(pd.DataFrame({'Author': lda_recs.index, 'Score': lda_recs.values}))

        with st.expander("ℹ️ How to interpret these results:"):
            st.markdown("""
            * **TF-IDF:** Finds reviewers using the *exact same keywords*.
            * **Embeddings:** Finds reviewers writing about the *same concepts*, even with different words.
            * **LDA:** Finds reviewers writing about similar *broad topics*.
            * **Scores:** Not comparable between methods. Higher score = higher similarity *within that method*.
            """)

# -----------------------------------------------------------------
# 5. UI: REVIEWER-REVIEWER SIMILARITY (Error handling added)
# -----------------------------------------------------------------
st.markdown("---")
st.header("🤝 Find Similar Reviewers")
st.write("Select an author to find others with similar overall expertise.")

# Make sure author_df is loaded before accessing it
if 'author_df' in locals() and not author_df.empty:
    all_authors = sorted(author_df['author'].unique())
    if "Unknown_Author" in all_authors:
         all_authors.remove("Unknown_Author") # Remove placeholder if used

    if not all_authors:
        st.warning("No authors found in the database to select from.")
    else:
        selected_author = st.selectbox("Select an author:", all_authors, key="similar_author_select") # Added key

        if st.button(f"Find Reviewers Similar to {selected_author}"):
            if selected_author:
                with st.spinner(f"Analyzing authors similar to {selected_author}..."):
                    similar_authors = get_similar_reviewers(selected_author, k=5)
                    if similar_authors.empty:
                        st.warning(f"Could not find similar reviewers for {selected_author}. The author might not be in the similarity matrix or the matrix is empty.")
                    else:
                        st.subheader(f"Top 5 Reviewers with Similar Expertise:")
                        st.dataframe(pd.DataFrame({
                            'Author': similar_authors.index,
                            'Similarity Score': similar_authors.values
                        }))
            else:
                st.warning("Please select an author.")
else:

    st.error("Author data could not be loaded. Cannot display reviewer similarity section.")
