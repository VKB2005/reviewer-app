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
import joblib
import tempfile

# --- Constants for filenames ---
AUTHOR_DB_FILE = 'author_database.parquet'
EMBEDDINGS_FILE = 'paper_embeddings.npy'
TFIDF_VEC_FILE = 'tfidf_vectorizer.joblib'
TFIDF_MAT_FILE = 'tfidf_matrix.joblib'
ST_MODEL_CACHE_DIR = './model_cache' 

# -----------------------------------------------------------------
# 1. SETUP & HELPER FUNCTIONS
# -----------------------------------------------------------------

poppler_install_path = r"C:\poppler\bin" 

@st.cache_data
def setup_nltk():
    resources = { 'corpora/stopwords': 'stopwords', 'tokenizers/punkt': 'punkt', 'tokenizers/punkt_tab': 'punkt_tab' }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{download_name}' found.")
        except LookupError:
            print(f"NLTK resource '{download_name}' not found. Downloading...")
            nltk.download(download_name, quiet=True)
setup_nltk()

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted:
                print(f"WARN: {os.path.basename(pdf_path)} is encrypted. Skipping.")
                return ""
            for page in doc: text += page.get_text()
        if text.strip(): return text
    except Exception as e_fitz:
        print(f"INFO: PyMuPDF failed on {os.path.basename(pdf_path)} ({e_fitz}). Trying OCR.")
        pass
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_install_path)
        if not images:
             print(f"WARN: pdf2image couldn't convert {os.path.basename(pdf_path)}. Skipping OCR.")
             return ""
        ocr_text = ""
        for img in images:
            try:
                ocr_text += pytesseract.image_to_string(img, lang='eng', timeout=30)
            except pytesseract.TesseractNotFoundError:
                 st.error("Tesseract is not installed or not in your PATH/packages.txt. OCR functionality will not work.")
                 print("ERROR: Tesseract not found during OCR.")
                 return ""
            except RuntimeError:
                 continue
            except Exception:
                 continue
        if ocr_text.strip():
             print(f"INFO: Successfully extracted text using OCR for {os.path.basename(pdf_path)}")
             return ocr_text
    except Exception as e_ocr:
        print(f"ERROR: OCR stage failed for {os.path.basename(pdf_path)} ({e_ocr}). Check Poppler/PDF corruption.")
        return ""
    print(f"WARN: Both PyMuPDF & OCR failed for {os.path.basename(pdf_path)}. Skipping.")
    return ""


def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
         return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(filtered_tokens) if filtered_tokens else ""

# --- *** NEW HELPER FUNCTION *** ---
def extract_authors_from_text(text_content, author_list):
    """A simple function to find known authors mentioned in a text."""
    authors_found = []
    # We check the first 2000 characters (usually where authors are)
    text_to_search = text_content[:2000].lower()
    
    for author in author_list:
        # Check if the author's name (lowercase) is in the text
        if author.lower() in text_to_search:
            authors_found.append(author)
            
    return list(set(authors_found)) # Return unique list

# -----------------------------------------------------------------
# 2. MODEL LOADING & PREPARATION
# -----------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and data files...")
def load_models_and_data():
    loaded_data = {}
    required_files = [AUTHOR_DB_FILE, EMBEDDINGS_FILE, TFIDF_VEC_FILE, TFIDF_MAT_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error(f"FATAL ERROR: Missing required data/model files: {', '.join(missing_files)}.")
        st.stop()
    
    try:
        author_df = pd.read_parquet(AUTHOR_DB_FILE)
        paper_embeddings = np.load(EMBEDDINGS_FILE)
        if author_df.empty:
            st.error(f"FATAL ERROR: Loaded '{AUTHOR_DB_FILE}' but it is empty.")
            st.stop()
        print(f"Loaded author_df with shape: {author_df.shape}")
        loaded_data['author_df'] = author_df
        loaded_data['paper_embeddings'] = paper_embeddings
        # --- Create the unique author list here ---
        loaded_data['all_authors_list'] = sorted(list(author_df['author'].dropna().unique()))
    except Exception as e:
        st.error(f"Error loading base data files: {e}")
        st.stop()

    try:
        loaded_data['tfidf_vectorizer'] = joblib.load(TFIDF_VEC_FILE)
        loaded_data['tfidf_matrix'] = joblib.load(TFIDF_MAT_FILE)
    except Exception as e:
        st.error(f"Error loading TF-IDF '.joblib' files: {e}")
        st.stop()

    try:
        if os.path.isdir(ST_MODEL_CACHE_DIR):
             loaded_data['model'] = SentenceTransformer(ST_MODEL_CACHE_DIR)
             print("Loaded SentenceTransformer from local cache.")
        else:
             print("Local model cache not found. Downloading/loading SentenceTransformer model...")
             loaded_data['model'] = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        st.stop()

    print("Building reviewer similarity matrix...")
    try:
        temp_author_df = loaded_data['author_df']
        temp_embeddings = loaded_data['paper_embeddings']
        min_len = min(len(temp_author_df), len(temp_embeddings))
        temp_author_df = temp_author_df.iloc[:min_len]
        temp_embeddings = temp_embeddings[:min_len]

        valid_indices = temp_author_df[temp_author_df['author'].notna()].index
        if len(valid_indices) == 0: raise ValueError("No valid authors.")

        author_vectors_df = (
            pd.DataFrame(temp_embeddings[valid_indices])
            .set_index(temp_author_df.loc[valid_indices, 'author'])
            .groupby(level=0)
            .mean()
        ).dropna()
        if author_vectors_df.empty: raise ValueError("Author vectors empty.")

        author_similarity_matrix = cosine_similarity(author_vectors_df)
        author_sim_df = pd.DataFrame(
            author_similarity_matrix,
            index=author_vectors_df.index,
            columns=author_vectors_df.index
        )
        loaded_data['author_sim_df'] = author_sim_df
        print("Reviewer similarity matrix built successfully.")
    except Exception as e:
        st.warning(f"Could not build author similarity matrix: {e}.")
        loaded_data['author_sim_df'] = pd.DataFrame()

    print("All models loaded!")
    return loaded_data

# --- Load all models ---
st.write("Loading all models...")
loaded_models_data = load_models_and_data()
author_df = loaded_models_data.get('author_df', pd.DataFrame())
paper_embeddings = loaded_models_data.get('paper_embeddings', np.array([]))
tfidf_vectorizer = loaded_models_data.get('tfidf_vectorizer')
tfidf_matrix = loaded_models_data.get('tfidf_matrix')
model = loaded_models_data.get('model')
author_sim_df = loaded_models_data.get('author_sim_df', pd.DataFrame())
all_authors_list = loaded_models_data.get('all_authors_list', []) # Get the author list

if author_df.empty:
    st.error("App initialization failed: Author database is empty.")
    st.stop()
else:
    st.success("All models loaded! ✅")

# -----------------------------------------------------------------
# 3. RECOMMENDATION FUNCTIONS (No changes)
# -----------------------------------------------------------------

def recommend_with_tfidf(processed_input, k=5):
    if not processed_input or tfidf_vectorizer is None or tfidf_matrix is None: return pd.Series(dtype='float64')
    try:
        input_vector = tfidf_vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        temp_df = pd.DataFrame({'author': author_df.iloc[:len(similarities)]['author'], 'similarity_score': similarities})
        top_authors = temp_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
        return top_authors.head(k)
    except Exception as e:
        print(f"Error in recommend_with_tfidf: {e}")
        return pd.Series(dtype='float64')

def recommend_with_embeddings(processed_input, k=5):
    if not processed_input or model is None or paper_embeddings.size == 0: return pd.Series(dtype='float64')
    try:
        input_embedding = model.encode([processed_input])
        embeddings_matrix = paper_embeddings
        if input_embedding.shape[1] != embeddings_matrix.shape[1]:
            print(f"ERROR: Embedding dimension mismatch. Input: {input_embedding.shape}, DB: {embeddings_matrix.shape}")
            return pd.Series(dtype='float64')
        similarities = cosine_similarity(input_embedding, embeddings_matrix).flatten()
        temp_df = pd.DataFrame({'author': author_df.iloc[:len(similarities)]['author'], 'similarity_score': similarities})
        top_authors = temp_df.groupby('author')['similarity_score'].max().sort_values(ascending=False)
        return top_authors.head(k)
    except Exception as e:
        print(f"Error in recommend_with_embeddings: {e}")
        return pd.Series(dtype='float64')

def get_similar_reviewers(author_name, k=5):
    if author_sim_df.empty or author_name not in author_sim_df:
        print(f"Author '{author_name}' not found or similarity matrix empty.")
        return pd.Series(dtype='float64')
    try:
        similar_scores = author_sim_df[author_name]
        top_similar = similar_scores.sort_values(ascending=False).drop(author_name)
        return top_similar.head(k)
    except Exception as e:
        print(f"Error in get_similar_reviewers for {author_name}: {e}")
        return pd.Series(dtype='float64')

# -----------------------------------------------------------------
# 4. UI: PAPER RECOMMENDATION (*** UPDATED FILTERING LOGIC ***)
# -----------------------------------------------------------------

st.title("👨‍🎓 Research Paper Reviewer Recommendation System")
st.write("Upload a PDF to find the best expert reviewers.")

# (Session state setup)
if 'k_value' not in st.session_state: st.session_state.k_value = 5
if 'file_bytes' not in st.session_state: st.session_state.file_bytes = None
if 'file_name' not in st.session_state: st.session_state.file_name = ""
if 'results' not in st.session_state: st.session_state.results = None
if 'last_uploaded_filename' not in st.session_state: st.session_state.last_uploaded_filename = ""

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
st.number_input("Number of reviewers to recommend (k):", min_value=3, max_value=20, key='k_value')
run_button = st.button("Find Reviewers for Paper")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.file_bytes = uploaded_file.getvalue()
        st.session_state.file_name = uploaded_file.name
        st.session_state.results = None
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.info(f"Loaded: **{st.session_state.file_name}**")

if run_button:
    if st.session_state.file_bytes is not None:
        with st.spinner("Processing paper and finding reviewers..."):
            tmp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(st.session_state.file_bytes)
                    tmp_file_path = tmp_file.name

                print(f"Processing temporary file: {tmp_file_path}")
                # We need the RAW text for author extraction
                input_text_raw = extract_text_from_pdf(tmp_file_path)

                if not input_text_raw:
                     st.error(f"Could not extract any text from the uploaded PDF: '{st.session_state.file_name}'.")
                     st.session_state.results = None
                else:
                    processed_input = preprocess_text(input_text_raw)
                    print(f"Length of processed text: {len(processed_input)}")

                    if not processed_input:
                         st.warning(f"Extracted text from '{st.session_state.file_name}' was empty after cleaning.")
                         st.session_state.results = (pd.Series(dtype='float64'), pd.Series(dtype='float64'))
                    else:
                        k = st.session_state.k_value
                        
                        # Get k+5 for robust filtering
                        tfidf_recs_raw = recommend_with_tfidf(processed_input, k=k+5)
                        embedding_recs_raw = recommend_with_embeddings(processed_input, k=k+5)

                        # --- *** NEW AUTHOR-NAME-BASED FILTERING *** ---
                        
                        # 1. Check for database match (score-based filter)
                        # We still use this for exact matches from our dataset
                        authors_to_exclude = list(embedding_recs_raw[embedding_recs_raw >= 0.98].index)
                        
                        # 2. Check for authors listed on the paper (name-based filter)
                        # This finds authors of NEW papers
                        listed_authors = extract_authors_from_text(input_text_raw, all_authors_list)
                        if listed_authors:
                            print(f"Found listed authors in PDF: {listed_authors}")
                            authors_to_exclude.extend(listed_authors)
                        
                        # Make the exclusion list unique
                        authors_to_exclude = list(set(authors_to_exclude))
                        if authors_to_exclude:
                            print(f"Final exclusion list: {authors_to_exclude}")
                        
                        # Apply the final exclusion list
                        tfidf_recs_filtered = tfidf_recs_raw[~tfidf_recs_raw.index.isin(authors_to_exclude)].head(k)
                        embedding_recs_filtered = embedding_recs_raw[~embedding_recs_raw.index.isin(authors_to_exclude)].head(k)
                        
                        st.session_state.results = (tfidf_recs_filtered, embedding_recs_filtered)

            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                        print(f"Removed temporary file: {tmp_file_path}")
                    except Exception as e_clean:
                        print(f"Error removing temp file {tmp_file_path}: {e_clean}")

    elif st.session_state.file_bytes is None:
        st.error("Please upload a PDF file first.")

# Display results
if st.session_state.results is not None:
    tfidf_recs, embedding_recs = st.session_state.results
    if not (tfidf_recs.empty and embedding_recs.empty) and st.session_state.file_name:
        st.success("Analysis complete!")
        st.subheader(f"Top {st.session_state.k_value} Recommended Reviewers for: **{st.session_state.file_name}**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### TF-IDF (Keywords)")
            st.dataframe(pd.DataFrame({'Author': tfidf_recs.index, 'Score': tfidf_recs.values}))
        with col2:
            st.markdown("##### Embeddings (Semantic)")
            st.dataframe(pd.DataFrame({'Author': embedding_recs.index, 'Score': embedding_recs.values}))
        with st.expander("ℹ️ How to interpret these results:"):
            st.markdown("""
            * **TF-IDF:** Finds reviewers using the *exact same keywords*.
            * **Embeddings:** Finds reviewers writing about the *same concepts*, even with different words.
            * **Scores:** Not comparable between methods. Higher score = higher similarity *within that method*.
            """)

# -----------------------------------------------------------------
# 5. UI: REVIEWER-REVIEWER SIMILARITY
# -----------------------------------------------------------------
st.markdown("---")
st.header("🤝 Find Similar Reviewers")
st.write("Select an author to find others with similar overall expertise.")

if not author_df.empty and 'author' in author_df.columns:
    all_authors = all_authors_list # Use the list from the loader
    if "Unknown_Author" in all_authors:
         all_authors.remove("Unknown_Author")

    if not all_authors:
        st.warning("No valid author names found in the database to select from.")
    else:
        selected_author = st.selectbox(
            "Select an author:",
            all_authors,
            key="similar_author_select",
            index=None,
            placeholder="Choose an author..."
        )

        if st.button(f"Find Reviewers Similar to Selected Author"):
            if selected_author:
                if author_sim_df.empty:
                     st.error("Reviewer similarity matrix could not be built during startup.")
                else:
                    with st.spinner(f"Analyzing authors similar to {selected_author}..."):
                        similar_authors = get_similar_reviewers(selected_author, k=5)
                        if similar_authors.empty:
                            st.warning(f"Could not find similar reviewers for {selected_author}.")
                        else:
                            st.subheader(f"Top 5 Reviewers with Similar Expertise:")
                            st.dataframe(pd.DataFrame({
                                'Author': similar_authors.index,
                                'Similarity Score': similar_authors.values
                            }))
            else:
                st.warning("Please select an author from the dropdown first.")
else:
    st.error("Author data is unavailable. Cannot display the 'Find Similar Reviewers' section.")
