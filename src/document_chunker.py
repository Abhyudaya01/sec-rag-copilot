import os
import pickle
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore
import faiss


# ------------------------------
# 1. Load filing text
# ------------------------------
def load_filing_text(filepath):
    """Read and return the full text of an SEC filing."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


# ------------------------------
# 2. Split document into chunks
# ------------------------------
def chunk_text(text, chunk_size=1800, chunk_overlap=200):
    """Split large text into overlapping chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text])


# ------------------------------
# 3. Embedding cache utilities
# ------------------------------
def save_embedding_cache(cache_path, cache_dict):
    """Save cached embeddings to disk."""
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_dict, f)

def load_embedding_cache(cache_path):
    """Load cached embeddings if available."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}


# ------------------------------
# 4. Batch embedding (local model)
# ------------------------------
def batch_embed_documents(embeddings_model, docs_to_embed, batch_size=5):
    """Generate embeddings for documents in small batches."""
    all_new_embeddings = []
    for i in range(0, len(docs_to_embed), batch_size):
        batch = docs_to_embed[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(
            [doc.page_content for doc in batch]
        )
        all_new_embeddings.extend(batch_embeddings)
    return all_new_embeddings


# ------------------------------
# 5. Create FAISS index (optimized)
# ------------------------------
def create_faiss_index(documents, index_path, cache_path="cache/embedding_cache.pkl"):
    """Create or update FAISS index using cached embeddings."""
    embedding_cache = load_embedding_cache(cache_path)

    # 🧠 Local embedding model (no API required)
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Determine which chunks need embedding
    docs_to_embed, indices_to_embed = [], []
    for i, doc in enumerate(documents):
        key = hash(doc.page_content)
        if key not in embedding_cache:
            docs_to_embed.append(doc)
            indices_to_embed.append(i)

    # Generate new embeddings if needed
    if docs_to_embed:
        print(f"Embedding {len(docs_to_embed)} new chunks...")
        new_embeddings = batch_embed_documents(
            embeddings_model, docs_to_embed, batch_size=5
        )
        for idx, embedding in zip(indices_to_embed, new_embeddings):
            embedding_cache[hash(documents[idx].page_content)] = embedding
        save_embedding_cache(cache_path, embedding_cache)
    else:
        print("All embeddings already cached.")

    # 🧠 Build FAISS index using cached embeddings
    print("Creating FAISS index...")

    # Initialize FAISS index
    dim = len(next(iter(embedding_cache.values())))  # embedding dimension
    index = faiss.IndexFlatL2(dim)

    # Add all cached embeddings
    emb_matrix = np.array(
        [embedding_cache[hash(doc.page_content)] for doc in documents]
    ).astype('float32')
    index.add(emb_matrix)

    # Build document store + mapping
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Create FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # Save index locally
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index created with {len(documents)} documents.")
    return vectorstore


# ------------------------------
# 6. Main execution
# ------------------------------
if __name__ == "__main__":
    filing_path = "data/processed/extracted_text.txt"
    index_path = "indexes/sec_filings_faiss"
    cache_path = "cache/embedding_cache.pkl"

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print("Loading filing text...")
    text = load_filing_text(filing_path)

    print("Chunking document...")
    documents = chunk_text(text, chunk_size=1800, chunk_overlap=200)
    print(f"Number of chunks: {len(documents)}")

    print("Generating embeddings and saving FAISS index...")
    result = create_faiss_index(documents, index_path, cache_path=cache_path)
    if result:
        print(f"✅ FAISS index saved to {index_path}")
    else:
        print("❌ FAISS index creation incomplete.")