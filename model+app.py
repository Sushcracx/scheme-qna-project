!pip install -q pandas sentence-transformers faiss-cpu
!pip install gradio
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

print("--- Initializing Scheme Retrieval App in Colab ---")

# --- Configuration ---
# IMPORTANT: Upload 'Data.json' to your Colab session first!
# This path assumes it's uploaded to the root /content/ directory.
file_path = '/content/Data/Data.json'
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
k_results_app = 5

# --- Global Variables ---
df = None
embedding_model = None
index = None
scheme_texts = None
setup_successful = False # Track if initialization worked

# --- Helper Function: Preprocessing ---
def combine_scheme_texts(row):
    """Combines relevant text fields from a scheme row."""
    row = row.fillna('')
    text_parts = []
    text_parts.append(f"Scheme Name: {row.get('Scheme Name', '')}")
    text_parts.append(f"Ministry: {row.get('Ministry', '')}")
    text_parts.append(f"Description: {row.get('Description', '')}")
    text_parts.append(f"Category: {row.get('Category', '')}")
    text_parts.append(f"Eligibility: {row.get('Eligibility', '')}")
    text_parts.append(f"Benefits: {row.get('Benefits', '')}")
    text_parts.append(f"Application Process: {row.get('Application Process', '')}")
    text_parts.append(f"Documents Required: {row.get('Documents', '')}") # Check column name
    if 'Target Audience' in row: text_parts.append(f"Target Audience: {row.get('Target Audience', '')}")
    return "\n".join(filter(None, text_parts))

# --- Setup Function (Run Once on Startup) ---
def initialize_retrieval_system():
    """Loads data, model, embeddings, and builds the index."""
    global df, embedding_model, index, scheme_texts, setup_successful
    print("\n--- Checking for Data File ---")
    if not os.path.exists(file_path):
        print(f"!!! ERROR: Data file not found at '{file_path}' !!!")
        setup_successful = False
        return # Stop initialization

    print("--- Loading and Preprocessing Data ---")
    try:
        df = pd.read_json(file_path)
        print(f"Loaded {len(df)} schemes.")
        df = df.fillna('')
        scheme_texts = [combine_scheme_texts(row) for _, row in df.iterrows()]
        print(f"Generated {len(scheme_texts)} combined texts.")
        if not scheme_texts:
             raise ValueError("No scheme texts generated.")

    except Exception as e:
        print(f"Error loading or preprocessing data: {e}")
        setup_successful = False
        return

    print(f"\n--- Loading Embedding Model ({embedding_model_name}) ---")
    start_time = time.time()
    embedding_model = SentenceTransformer(embedding_model_name)
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f} seconds.")

    print("\n--- Generating Embeddings (this may take a moment) ---")
    start_time = time.time()
    scheme_embeddings = embedding_model.encode(scheme_texts, show_progress_bar=True, convert_to_numpy=True)
    end_time = time.time()
    print(f"Embeddings generated. Shape: {scheme_embeddings.shape}. Time: {end_time - start_time:.2f}s")

    print("\n--- Building FAISS Index ---")
    start_time = time.time()
    faiss.normalize_L2(scheme_embeddings)
    d = scheme_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(scheme_embeddings)
    end_time = time.time()
    print(f"FAISS index built. Indexed {index.ntotal} vectors. Time: {end_time - start_time:.2f}s")

    print("\n--- Initialization Complete ---")
    setup_successful = True # Mark setup as successful

# --- Core Retrieval Logic (Interface Function) ---
def find_schemes_interface(query):
    """Takes a query string and returns formatted results string."""
    global df, embedding_model, index, scheme_texts, setup_successful

    if not setup_successful:
         return "ERROR: System initialization failed. Please check the logs above."
    if not query:
        return "Please enter a query."
    if index is None or embedding_model is None or df is None or scheme_texts is None:
         return "ERROR: System components not loaded."
    if index.ntotal == 0:
        return "Warning: FAISS index is empty."

    print(f"\nReceived query: '{query}'")
    start_time = time.time()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    try:
        D, I = index.search(query_embedding, k_results_app)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return f"Error during search: {e}"

    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.4f} seconds.")

    # Format Results
    results_string = f"Found {I.shape[1]} potential matches for '{query}' (Search time: {search_time:.2f}s):\n\n"
    if I.size > 0:
        for i in range(I.shape[1]):
            scheme_index = I[0][i]
            similarity_score = D[0][i]
            if scheme_index < 0 or scheme_index >= len(df):
                results_string += f"--- Result {i+1} ---\nInvalid index: {scheme_index}\n\n"
                continue
            original_scheme_info = df.iloc[scheme_index]
            results_string += f"--- Result {i+1} ---\n"
            results_string += f"**Scheme Name:** {original_scheme_info.get('Scheme Name', 'N/A')}\n"
            results_string += f"**Similarity Score:** {similarity_score:.4f}\n"
            results_string += f"**Ministry:** {original_scheme_info.get('Ministry', 'N/A')}\n"
            results_string += f"**Category:** {original_scheme_info.get('Category', 'N/A')}\n"
            description = original_scheme_info.get('Description', 'N/A')
            description_snippet = (description[:300] + '...') if len(description) > 300 else description
            results_string += f"**Description:** {description_snippet}\n"
            results_string += f"**Source URL:** {original_scheme_info.get('Source URL', 'N/A')}\n\n"
    else:
        results_string += "No relevant schemes found."
    return results_string

# --- Run Setup ---
initialize_retrieval_system()

# --- Create and Launch Gradio Interface ---
if setup_successful:
    print("\n--- Launching Gradio Interface ---")
    print("Wait for the public URL link to appear below...")
    iface = gr.Interface(
        fn=find_schemes_interface,
        inputs=gr.Textbox(lines=3, placeholder="Enter your question about government schemes here... e.g., 'schemes for farmers', 'financial help for students'"),
        outputs=gr.Markdown(label="Retrieved Schemes"),
        title="Government Scheme Information Retrieval",
        description="Ask a question to find relevant government. Results are ranked by relevance.",
        allow_flagging='never',
        examples=[
            ["What are the schemes for farmers?"],
            ["Financial assistance for higher education"],
            ["Help for starting a small business"],
            ["Housing schemes in rural areas"],
            ["Sukanya Samriddhi Yojana details"]
        ]
    )

    # Use share=True to get a public link in Colab
    iface.launch(share=True, debug=False) # debug=True can provide more logs if needed
else:
    print("\n--- Gradio interface did not launch due to initialization errors. ---")
    print("Please check the error messages above, ensure the data file was uploaded correctly, and try running the cell again.")