
!pip install -q pandas sentence-transformers faiss-cpu
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time
import os

file_path = '/content/Data/Data.json'
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2' # Small & efficient model
k_results = 5 # Number of results to retrieve for a query

# --- Load Data ---
print(f"\n--- Loading data from {file_path} ---")
if not os.path.exists(file_path):
    print(f"ERROR: File not found at {file_path}.")
    print("Please upload the file to your Colab session's root directory and try again.")
    # Stop execution if file not found
    raise FileNotFoundError(f"{file_path} not found. Please upload the file.")
else:
    try:
        # Load the JSON data into a pandas DataFrame
        df = pd.read_json(file_path)
        print(f"Successfully loaded data. Number of schemes: {len(df)}")

        # Display basic info to verify
        print("\nFirst 5 rows of the dataframe:")
        print(df.head())
        print("\nColumns in the dataframe:")
        print(df.columns.tolist())

    except Exception as e:
        print(f"An error occurred while loading or processing the JSON file: {e}")
        raise e # Stop execution on error

# Step 3: Preprocess and Combine Text Data
print("\n--- Preprocessing Data ---")

# Fill potential missing values (NaN) with empty strings before combining
df = df.fillna('')

def combine_scheme_texts(row):
    """
    Combines relevant text fields from a scheme row into a single string.
    Adjust the fields included here based on your JSON structure and what's important for retrieval.
    """
    # Check which columns exist before trying to access them
    text_parts = []
    if 'Scheme Name' in row: text_parts.append(f"Scheme Name: {row['Scheme Name']}")
    if 'Ministry' in row: text_parts.append(f"Ministry: {row['Ministry']}")
    if 'Description' in row: text_parts.append(f"Description: {row['Description']}")
    if 'Category' in row: text_parts.append(f"Category: {row['Category']}")
    if 'Eligibility' in row: text_parts.append(f"Eligibility: {row['Eligibility']}")
    if 'Benefits' in row: text_parts.append(f"Benefits: {row['Benefits']}")
    if 'Application Process' in row: text_parts.append(f"Application Process: {row['Application Process']}")
    if 'Documents Required' in row: text_parts.append(f"Documents Required: {row['Documents Required']}") # Check column name carefully
    if 'Target Audience' in row: text_parts.append(f"Target Audience: {row['Target Audience']}")

    return "\n".join(text_parts) # Join parts with newline for readability

# Create a list of combined texts
scheme_texts = [combine_scheme_texts(row) for index, row in df.iterrows()]

# Store the original index for later reference
scheme_ids = df.index.tolist()

print(f"Created {len(scheme_texts)} combined text documents for schemes.")
if scheme_texts:
  print("\nExample combined text for the first scheme (first 1000 chars):")
  print(scheme_texts[0][:1000] + "...")
else:
    print("Warning: No scheme texts were generated. Check the combine_scheme_texts function and DataFrame columns.")

# Step 4: Embedding Generation
print(f"\n--- Loading Embedding Model ({embedding_model_name}) ---")
start_time = time.time()
embedding_model = SentenceTransformer(embedding_model_name)
end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")

print("\n--- Generating Embeddings ---")
# Check if there are texts to embed
if not scheme_texts:
    print("Error: Cannot generate embeddings because scheme_texts list is empty.")
    raise ValueError("No text data available for embedding.")

start_time = time.time()
# Generate embeddings for all scheme texts
scheme_embeddings = embedding_model.encode(scheme_texts, show_progress_bar=True, convert_to_numpy=True)
end_time = time.time()

print(f"Embeddings generated. Shape: {scheme_embeddings.shape}")
print(f"Time taken for embedding: {end_time - start_time:.2f} seconds")

# Step 5: Vector Indexing using FAISS
print("\n--- Building FAISS Index ---")
start_time = time.time()

# Get the dimension of the embeddings
d = scheme_embeddings.shape[1]

# Using IndexFlatIP for Inner Product (Cosine Similarity after normalization)
# Normalize the embeddings L2 norm -> vectors of norm 1
faiss.normalize_L2(scheme_embeddings)

# Create the index
index = faiss.IndexFlatIP(d)

# Add the normalized scheme embeddings to the index
index.add(scheme_embeddings)
end_time = time.time()

print(f"FAISS index built. Index type: IndexFlatIP")
print(f"Number of vectors indexed: {index.ntotal}")
print(f"Time taken for indexing: {end_time - start_time:.2f} seconds")

# Step 6: Retrieval Function
def retrieve_schemes(query, k=5):
    """
    Embeds the query and retrieves the top-k most similar schemes from the FAISS index.
    Args:
        query (str): The user's natural language query.
        k (int): The number of top results to retrieve.
    Returns:
        list: A list of dictionaries, each containing info about a retrieved scheme.
    """
    if not query:
        print("Warning: Empty query provided.")
        return []
    if index.ntotal == 0:
        print("Warning: FAISS index is empty. Cannot perform search.")
        return []

    print(f"\n--- Retrieving Top {k} Schemes for Query: '{query}' ---")
    start_time = time.time()

    # 1. Embed the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # 2. Normalize the query embedding (important for IndexFlatIP)
    faiss.normalize_L2(query_embedding)

    # 3. Search the index
    # D: distances (inner product scores - higher is better for IP)
    # I: indices of the nearest neighbors in the original dataset
    try:
        D, I = index.search(query_embedding, k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds.")

    # 4. Format and return results
    results = []
    if I.size > 0:
        for i in range(I.shape[1]): # Iterate through the k results found
            scheme_index = I[0][i]
            similarity_score = D[0][i]

            # Ensure index is valid
            if scheme_index < 0 or scheme_index >= len(df):
                print(f"Warning: Invalid index {scheme_index} retrieved from FAISS. Skipping.")
                continue

            # Retrieve original data using the index
            original_scheme_info = df.iloc[scheme_index]

            # Get the text that was actually indexed
            retrieved_text = scheme_texts[scheme_index] if scheme_index < len(scheme_texts) else "Error: Text not found"

            results.append({
                'rank': i + 1,
                'scheme_index': int(scheme_index), # Original DataFrame index
                'similarity_score': float(similarity_score), # Higher is more similar for IP
                'scheme_name': original_scheme_info.get('Scheme Name', 'N/A'),
                'ministry': original_scheme_info.get('Ministry', 'N/A'),
                'description': original_scheme_info.get('Description', 'N/A')[:300] + "...", # Truncate for display
                'category': original_scheme_info.get('Category', 'N/A'),
                'source_url': original_scheme_info.get('Source URL', 'N/A'),
                # 'full_text_retrieved': retrieved_text # Uncomment if you want to see the full text used for retrieval
            })
    else:
        print("No results found for the query.")

    return results

# Step 7: Example Usage
print("\n--- Example Retrieval ---")

# Example query based on your project description
# Note: Specificity like 'in Maharashtra' might be hard if location isn't consistently in the combined text.
# Let's try some general and slightly specific queries.
queries = [
    "What financial assistance schemes are available for students?",
    "Any schemes for farmers?",
    "Schemes related to housing for rural areas",
    "Help for small businesses",
    "What is Pradhan Mantri Jan Dhan Yojana?" # Querying a specific scheme name
]

# Run retrieval for each query
for sample_query in queries:
    retrieved_results = retrieve_schemes(sample_query, k=k_results)

    print(f"\n--- Results for Query: '{sample_query}' ---")
    if retrieved_results:
        for result in retrieved_results:
            print(f"\nRank: {result['rank']}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
            print(f"  Scheme Name: {result['scheme_name']}")
            print(f"  Ministry: {result['ministry']}")
            print(f"  Category: {result['category']}")
            print(f"  Description (preview): {result['description']}")
            print(f"  Source URL: {result['source_url']}")
            # print(f"  Retrieved Index: {result['scheme_index']}") # For debugging
    else:
        print("  No relevant schemes found.")

print("\n--- End of Initial Retrieval Setup ---")

import pandas as pd

print("\n--- Starting Retrieval Accuracy Evaluation ---")

# --- !! IMPORTANT !! ---
# YOU MUST DEFINE YOUR 20-30+ EVALUATION QUESTIONS AND GROUND TRUTH HERE
# Use the exact 'Scheme Name' from your DataFrame as the ground truth.
# Structure: { "query_id": {"query": "Your question?", "ground_truth": ["Exact Scheme Name 1", ...]}, ... }

evaluation_data = {
    # --- Start replacing/adding your 20-30+ questions/answers below ---
    "q1_student_finance": {
        "query": "What financial assistance schemes are available for students?",
        "ground_truth": ["Central Sector Scheme of Scholarship for College and University Students"] # Replace with ACTUAL scheme name(s) from your data
    },
    "q2_farmer_general": {
        "query": "Any schemes for farmers?",
        "ground_truth": ["Pradhan Mantri Kisan Samman Nidhi", "Pradhan Mantri Fasal Bima Yojana"] # Replace with ACTUAL scheme name(s)
    },
    "q3_housing_rural": {
        "query": "Schemes related to housing for rural areas",
        "ground_truth": ["Pradhan Mantri Awaas Yojana - Gramin"] # Replace with ACTUAL scheme name(s)
    },
    "q4_small_business": {
        "query": "Help for small businesses",
        "ground_truth": ["Pradhan Mantri MUDRA Yojana (PMMY)"] # Replace with ACTUAL scheme name(s)
    },
    "q5_pmjdy": {
        "query": "What is Pradhan Mantri Jan Dhan Yojana?",
        "ground_truth": ["Pradhan Mantri Jan Dhan Yojana (PMJDY)"] # Replace if name slightly different
    },
    "q6_girl_child": {
         "query": "Schemes for the girl child",
         "ground_truth": ["Sukanya Samriddhi Yojana", "Beti Bachao Beti Padhao Scheme"] # Example - replace with actuals
    },
     "q7_skill_dev": {
         "query": "Training for youth skills",
         "ground_truth": ["Pradhan Mantri Kaushal Vikas Yojana (PMKVY)"] # Replace with actual
     },
    "q8_widow_pension": {
        "query": "Is there a pension for widows?",
        "ground_truth": ["Indira Gandhi National Widow Pension Scheme"] # Replace with actual
    },
    "q9_health_insurance": {
        "query": "Government health insurance",
        "ground_truth": ["Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB-PMJAY)"] # Replace with actual
    },
    "q10_disability_aid": {
        "query": "Assistance for persons with disabilities",
        "ground_truth": ["Accessible India Campaign (Sugamya Bharat Abhiyan)"] # Replace with actual, check scheme type
    },
    "q11_senior_citizen_saving": {
        "query": "Savings scheme for senior citizens",
        "ground_truth": ["Senior Citizen Savings Scheme (SCSS)"] # Replace with actual
    },
    "q12_crop_insurance": {
        "query": "Insurance for crops",
        "ground_truth": ["Pradhan Mantri Fasal Bima Yojana"] # Replace with actual
    },
    "q13_msme_loan": {
        "query": "Loan for MSME sector",
        "ground_truth": ["Credit Guarantee Fund Scheme for Micro and Small Enterprises (CGFMSE)"] # Replace with actual
    },
    "q14_clean_india": {
        "query": "Schemes related to Clean India mission",
        "ground_truth": ["Swachh Bharat Mission (SBM)"] # Replace with actual
    },
    "q15_digital_india": {
        "query": "Programs under Digital India",
        "ground_truth": ["DigiLocker", "MyGov.in"] # Replace with actual schemes/programs if listed
    },
    "q16_maternity_benefit": {
        "query": "Benefits for pregnant women",
        "ground_truth": ["Pradhan Mantri Matru Vandana Yojana (PMMVY)"] # Replace with actual
    },
    "q17_pension_unorganized": {
        "query": "Pension for unorganized sector workers",
        "ground_truth": ["Pradhan Mantri Shram Yogi Maan-dhan (PM-SYM)"] # Replace with actual
    },
    "q18_rural_employment": {
        "query": "Rural employment guarantee",
        "ground_truth": ["Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA)"] # Replace with actual
    },
    "q19_startup_funding": {
        "query": "Funding for startup companies",
        "ground_truth": ["Startup India Seed Fund Scheme"] # Replace with actual
    },
    "q20_lpg_connection": {
        "query": "Scheme for free LPG connection",
        "ground_truth": ["Pradhan Mantri Ujjwala Yojana (PMUY)"] # Replace with actual
    }
    # ====> Add more unique questions to reach 25-30 <====

} # <--- Ensure this closing brace is present

# --- Evaluation Parameters ---
# How many top results to check for a correct answer?
k_eval = 5 # Common choices: 1, 3, 5

# --- Check if required variables exist from previous cell ---
if 'retrieve_schemes' not in globals() or 'df' not in globals() or 'index' not in globals():
     print("\nERROR: Setup variables (retrieve_schemes, df, index) not found.")
     print("Please ensure the previous cell with the retrieval setup ran successfully.")
else:
    # --- Evaluation Metrics Storage ---
    results_summary = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_k = 0 # Hits within top k_eval
    total_reciprocal_rank = 0.0

    # --- Run Evaluation Loop ---
    num_queries = len(evaluation_data)
    if num_queries < 10: # Add a warning if the set is small
        print(f"\nWARNING: Evaluating with only {num_queries} questions. Consider adding more (20-30 recommended) for reliable results.")
    elif num_queries == 0:
        print("\nERROR: evaluation_data dictionary is empty. Please add evaluation questions and ground truth.")
    else:
        print(f"\nEvaluating {num_queries} queries, checking top {k_eval} results...")

        for query_id, data in evaluation_data.items():
            query = data["query"]
            ground_truth_names = data["ground_truth"] # List of expected scheme names

            # Use the retrieve_schemes function defined earlier in your code
            retrieved_results = retrieve_schemes(query, k=k_eval) # Use k_eval here

            retrieved_names = [res['scheme_name'] for res in retrieved_results]

            # --- Calculate Metrics for this query ---
            is_hit_at_1 = False
            is_hit_at_3 = False
            is_hit_at_k = False
            found_rank = -1 # Rank of the first ground truth item found (1-based)

            for rank, name in enumerate(retrieved_names, 1):
                if name in ground_truth_names:
                    is_hit_at_k = True # Found within top k
                    if rank == 1: is_hit_at_1 = True
                    if rank <= 3: is_hit_at_3 = True
                    if found_rank == -1: found_rank = rank
                    # break # Uncomment if you only care that *at least one* correct item was found

            # Update overall counters
            if is_hit_at_1: hits_at_1 += 1
            if is_hit_at_3: hits_at_3 += 1
            if is_hit_at_k: hits_at_k += 1

            reciprocal_rank = 1.0 / found_rank if found_rank != -1 else 0.0
            total_reciprocal_rank += reciprocal_rank

        print("\n--- Evaluation Complete ---")

        # --- Calculate Final Metrics ---
        if num_queries > 0:
            recall_at_1 = hits_at_1 / num_queries
            recall_at_3 = hits_at_3 / num_queries
            recall_at_k = hits_at_k / num_queries
            mean_reciprocal_rank = total_reciprocal_rank / num_queries
        else:
            recall_at_1 = recall_at_3 = recall_at_k = mean_reciprocal_rank = 0.0

        # --- Display Accuracy Results as Percentages ---
        print("\n--- Overall Retrieval Accuracy ---")
        print(f"Total Queries Evaluated: {num_queries}")
        print(f"K (results considered per query): {k_eval}")
        # Format Recall@k as percentages
        print(f"Accuracy@1 (Recall@1): {recall_at_1 * 100:.2f}% ({hits_at_1}/{num_queries})")
        print(f"Accuracy@3 (Recall@3): {recall_at_3 * 100:.2f}% ({hits_at_3}/{num_queries})")
        print(f"Accuracy@{k_eval} (Recall@{k_eval}): {recall_at_k * 100:.2f}% ({hits_at_k}/{num_queries})")
        print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.4f}")