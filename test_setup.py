import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Test Gemini connection
print("Testing Gemini API connection...")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0
    )
    
    # Test query
    response = llm.invoke("Say 'Setup successful!' if you can read this.")
    print(f"\n✅ Success! Gemini responded: {response.content}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Check your API key in the .env file")