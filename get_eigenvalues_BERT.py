# Extract input components
input_ids = preprocessed_inputs["input_word_ids"]
input_mask = preprocessed_inputs["input_mask"]
segment_ids = preprocessed_inputs["input_type_ids"]

print("Preprocessing Complete.")
print(f"Input IDs shape: {input_ids.shape}")
print(f"Input Mask shape: {input_mask.shape}")
print(f"Segment IDs shape: {segment_ids.shape}")

# 4. Extract BERT Embeddings ([CLS] Token)
def get_bert_embeddings(bert_layer, input_ids, input_mask, segment_ids):
    """
    Obtains BERT embeddings for the [CLS] token.

    Parameters:
    - bert_layer: TensorFlow Hub KerasLayer for BERT.
    - input_ids: Tensor of input IDs.
    - input_mask: Tensor of input masks.
    - segment_ids: Tensor of segment IDs.

    Returns:
    - cls_embeddings: NumPy array of [CLS] token embeddings.
    """
    # Pass inputs through BERT
    bert_outputs = bert_layer({
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    })
    
    # Extract the pooled_output ([CLS] token)
    cls_embeddings = bert_outputs['pooled_output'].numpy()
    return cls_embeddings

bert_embeddings = get_bert_embeddings(bert_model, input_ids, input_mask, segment_ids)
print(f"BERT Embeddings Shape: {bert_embeddings.shape}")  # Expected: (num_documents, 768)

# 5. Vectorize the Dataset Using TF-IDF
def vectorize_tfidf(texts, max_features=1000):
    """
    Vectorizes texts using TF-IDF.

    Parameters:
    - texts: List of strings.
    - max_features: Maximum number of features for TF-IDF.

    Returns:
    - tfidf_matrix: Sparse matrix of TF-IDF features.
    - tfidf_vectorizer: Fitted TfidfVectorizer object.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer

tfidf_matrix, tfidf_vectorizer = vectorize_tfidf(documents, max_features=1000)
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")  # Expected: (num_documents, max_features)

# 6. Compute Eigenvalues from Covariance Matrices
def compute_eigenvalues(matrix):
    """
    Computes eigenvalues from the covariance matrix of the input data.

    Parameters:
    - matrix: NumPy array or sparse matrix of shape (n_samples, n_features).

    Returns:
    - eigenvalues: 1D NumPy array of eigenvalues sorted in descending order.
    """
    # Convert sparse matrix to dense if necessary
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    
    # Standardize the data (mean=0)
    scaler = StandardScaler(with_mean=True, with_std=False)
    matrix_centered = scaler.fit_transform(matrix)
    
    # Compute covariance matrix
    covariance_matrix = np.cov(matrix_centered, rowvar=False)
    
    # Compute eigenvalues (ascending order)
    eigenvalues = eigvalsh(covariance_matrix)
    
    # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[::-1]
    
    return eigenvalues

# Compute eigenvalues for TF-IDF
tfidf_eigenvalues = compute_eigenvalues(tfidf_matrix)
print("TF-IDF Eigenvalues Computed.")

# Compute eigenvalues for BERT Embeddings
bert_eigenvalues = compute_eigenvalues(bert_embeddings)
print("BERT Eigenvalues Computed.")

# 7. Visualize and Compare the Eigenvalues
def plot_eigenvalues(tfidf_evals, bert_evals, num_eigenvalues=50):
    """
    Plots the top eigenvalues for TF-IDF and BERT embeddings.

    Parameters:
    - tfidf_evals: 1D array of TF-IDF eigenvalues.
    - bert_evals: 1D array of BERT eigenvalues.
    - num_eigenvalues: Number of top eigenvalues to plot.
    """
    plt.figure(figsize=(14, 7))
    
    plt.plot(tfidf_evals[:num_eigenvalues], label='TF-IDF', marker='o')
    plt.plot(bert_evals[:num_eigenvalues], label='BERT', marker='x')
    
    plt.title('Comparison of Eigenvalues: TF-IDF vs BERT')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_eigenvalues(tfidf_eigenvalues, bert_eigenvalues, num_eigenvalues=50)

# 8. Additional Comparative Analysis (Optional)
# 8A. Cumulative Explained Variance
def plot_cumulative_variance(tfidf_evals, bert_evals):
    """
    Plots cumulative explained variance for TF-IDF and BERT embeddings.

    Parameters:
    - tfidf_evals: 1D array of TF-IDF eigenvalues.
    - bert_evals: 1D array of BERT eigenvalues.
    """
    tfidf_cum_var = np.cumsum(tfidf_evals) / np.sum(tfidf_evals)
    bert_cum_var = np.cumsum(bert_evals) / np.sum(bert_evals)
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(tfidf_cum_var, label='TF-IDF Cumulative Explained Variance')
    plt.plot(bert_cum_var, label='BERT Cumulative Explained Variance')
    
    plt.title('Cumulative Explained Variance Comparison')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_cumulative_variance(tfidf_eigenvalues, bert_eigenvalues)

