import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load .env to get your Gemini API key
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# 1. Read processed SEC text
def load_filing_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# 2. Chunk the document
def chunk_text(text, chunk_size=1800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text])  # returns list of Document objects

# 3. Embed and store in FAISS
def create_faiss_index(documents, index_path):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

if __name__ == "__main__":
    filing_path = "data/processed/extracted_text.txt"   # Input your cleaned extracted text path
    index_path = "indexes/sec_filings_faiss"            # Output FAISS index path

    print("Loading filing text...")
    text = load_filing_text(filing_path)

    print("Chunking document...")
    documents = chunk_text(text, chunk_size=1800, chunk_overlap=200)
    print(f"Number of chunks: {len(documents)}")

    print("Generating embeddings and saving FAISS index...")
    create_faiss_index(documents, index_path)
    print(f"FAISS index saved to {index_path}")
