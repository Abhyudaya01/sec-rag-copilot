"""
Gemini configuration for SEC RAG copilot
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def get_gemini_llm(model="gemini-1.5-flash", temperature=0):  # ← Changed from gemini-2.0-flash-exp
    """Initialize Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=temperature,
        convert_system_message_to_human=True
    )

def get_gemini_embeddings():
    """Initialize Gemini embeddings"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

if __name__ == "__main__":
    # Test the setup
    print("Testing Gemini setup...")
    llm = get_gemini_llm()
    response = llm.invoke("Hello, Gemini!")
    print(f"✅ LLM working: {response.content}")
    
    embeddings = get_gemini_embeddings()
    test_embedding = embeddings.embed_query("test")
    print(f"✅ Embeddings working: {len(test_embedding)} dimensions")