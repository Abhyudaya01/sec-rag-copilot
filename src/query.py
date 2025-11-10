import os
import asyncio
import logging
import time
import re
import functools
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Optional, List
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Configurations ---
FAISS_INDEX_PATH = "indexes/sec_filings_faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash" 
DEFAULT_TOP_K = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------
# Custom Gemini LLM Wrapper
# -------------------------------------------
class GeminiLLM(LLM):
    """Custom Gemini LLM wrapper for LangChain"""
    
    model_name: str = GEMINI_MODEL_NAME
    temperature: float = 0
    model: object = None
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file. Please add:\n"
                "GOOGLE_API_KEY=your-api-key-here"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)
        logging.info(f"Initialized Gemini model: {self.model_name}")
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error calling Gemini: {e}")
            raise
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)

# -------------------------------------------
# Embeddings Initialization with Caching
# -------------------------------------------
@functools.lru_cache(maxsize=1)
def get_embeddings():
    """Initialize and cache the embedding model to avoid reloading."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# -------------------------------------------
# Initialize Gemini LLM
# -------------------------------------------
def get_llm():
    """Initialize custom Gemini LLM"""
    return GeminiLLM()

# -------------------------------------------
# Load FAISS Retriever
# -------------------------------------------
def load_retriever(k=DEFAULT_TOP_K):
    """
    Load FAISS index and return a retriever with dynamic top-k.
    """
    embeddings = get_embeddings()
    
    # Load FAISS index - safe because we created it ourselves
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Return retriever without compression (simpler and faster)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# -------------------------------------------
# Chain-of-Thought Prompt Templates for SEC Filings
# -------------------------------------------

# General SEC Analysis CoT Prompt
GENERAL_SEC_COT = """You are an expert financial analyst AI. Using the following SEC filing excerpts, answer the question with step-by-step reasoning and cite the specific chunks used.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Identify relevant sections: Review the provided excerpts from the Business Overview, Risk Factors, MD&A, Financial Statements, or other sections relevant to the question.
2. Extract key data points: Pull specific financial metrics, trends, or statements that address the question directly.
3. Analyze and compare: Compare figures to prior periods or note significant changes, and identify management's explanations for trends.
4. Assess implications: Consider what these findings mean for the company's financial health, risks, or strategic position.
5. Cite sources: Reference the specific document chunks used in your analysis.

Final answer: [Provide a clear, concise answer with citations]
"""

# Risk Analysis CoT Prompt
RISK_ANALYSIS_COT = """You are an expert financial risk analyst. Using the SEC filing excerpts below, identify and analyze risk factors with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Identify risk categories: Review Risk Factors section (Item 1A) and categorize risks (operational, financial, regulatory, market, etc.).
2. Extract specific risks: List the key risks mentioned with exact language from the filing.
3. Assess severity: Evaluate the potential impact and likelihood of each risk based on management's discussion.
4. Compare to industry: Note any risks that are unique to this company versus common industry risks.
5. Cite sources: Reference specific chunks for each identified risk.

Final answer: [List main risks with severity assessment and citations]
"""

# Financial Performance CoT Prompt
FINANCIAL_PERFORMANCE_COT = """You are an expert in financial statement analysis. Using the SEC filing excerpts, analyze financial performance with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Identify financial metrics: Extract revenue, net income, cash flow, margins, and other key metrics from the financial statements.
2. Calculate changes: Compare current period to prior periods and calculate percentage changes.
3. Identify drivers: Review MD&A to understand what management says drove these changes.
4. Assess quality of earnings: Note any one-time items, non-recurring charges, or accounting policy changes.
5. Cite sources: Reference specific chunks for all financial data and management commentary.

Final answer: [Provide financial performance summary with trends and citations]
"""

# Liquidity & Solvency CoT Prompt
LIQUIDITY_SOLVENCY_COT = """You are an expert in corporate liquidity and solvency analysis. Using the SEC filing excerpts, assess the company's financial position.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Extract balance sheet data: Identify current assets, current liabilities, total debt, and shareholders' equity.
2. Calculate key ratios: Compute current ratio, quick ratio, debt-to-equity ratio, and interest coverage ratio.
3. Review cash flow: Analyze operating cash flow, free cash flow, and cash position trends.
4. Management discussion: Review MD&A for management's assessment of liquidity and capital resources.
5. Cite sources: Reference specific chunks for all financial data.

Final answer: [Provide liquidity and solvency assessment with ratios and citations]
"""

# Business Operations CoT Prompt
BUSINESS_OPERATIONS_COT = """You are an expert business analyst. Using the SEC filing excerpts, explain the company's business model and operations.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Business description: Review Item 1 "Business" section to understand what the company does.
2. Products and services: Identify main offerings, target markets, and customer segments.
3. Competitive position: Note the company's competitive advantages and market position.
4. Segment analysis: If applicable, break down performance by business segment or geography.
5. Cite sources: Reference specific chunks for business descriptions.

Final answer: [Provide business overview with key operational insights and citations]
"""

# Regulatory & Legal CoT Prompt
REGULATORY_LEGAL_COT = """You are an expert in corporate legal and regulatory matters. Using the SEC filing excerpts, analyze legal and compliance issues.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Identify legal proceedings: Review Item 3 "Legal Proceedings" for pending litigation or regulatory matters.
2. Assess materiality: Evaluate the potential financial impact and likelihood of adverse outcomes.
3. Regulatory environment: Note key regulations affecting the business and any compliance challenges.
4. Risk implications: Consider how legal/regulatory issues might affect future operations or financial results.
5. Cite sources: Reference specific chunks for all legal and regulatory information.

Final answer: [Summarize legal and regulatory position with risk assessment and citations]
"""

# -------------------------------------------
# Prompt Selection Logic
# -------------------------------------------
def select_prompt_template(question: str) -> PromptTemplate:
    """
    Select the most appropriate CoT prompt based on question keywords.
    Falls back to general SEC analysis prompt if no specific match.
    """
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['risk', 'risks', 'threat', 'challenge', 'exposure']):
        template = RISK_ANALYSIS_COT
        logging.info("Selected: Risk Analysis CoT Prompt")
    elif any(word in question_lower for word in ['revenue', 'profit', 'earnings', 'income', 'performance', 'sales', 'margin']):
        template = FINANCIAL_PERFORMANCE_COT
        logging.info("Selected: Financial Performance CoT Prompt")
    elif any(word in question_lower for word in ['liquidity', 'cash', 'debt', 'solvency', 'assets', 'liabilities', 'balance sheet']):
        template = LIQUIDITY_SOLVENCY_COT
        logging.info("Selected: Liquidity & Solvency CoT Prompt")
    elif any(word in question_lower for word in ['business', 'operations', 'products', 'services', 'market', 'industry', 'competitive']):
        template = BUSINESS_OPERATIONS_COT
        logging.info("Selected: Business Operations CoT Prompt")
    elif any(word in question_lower for word in ['legal', 'lawsuit', 'litigation', 'regulatory', 'compliance', 'violation']):
        template = REGULATORY_LEGAL_COT
        logging.info("Selected: Regulatory & Legal CoT Prompt")
    else:
        template = GENERAL_SEC_COT
        logging.info("Selected: General SEC Analysis CoT Prompt")
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# -------------------------------------------
# Query Embedding Cache
# -------------------------------------------
query_embedding_cache = {}

def get_query_embedding(query, embeddings_model):
    """Cache query embeddings to speed up repeated questions."""
    if query in query_embedding_cache:
        logging.info("Using cached query embedding")
        return query_embedding_cache[query]
    embedding = embeddings_model.embed_query(query)
    query_embedding_cache[query] = embedding
    return embedding

# -------------------------------------------
# Input Validation
# -------------------------------------------
def validate_input(query):
    """Validate and sanitize user input."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if len(query) < 3:
        raise ValueError("Query is too short. Please provide more detail.")
    if len(query) > 500:
        raise ValueError("Query is too long. Please keep it under 500 characters.")
    # Basic sanitization - remove potentially harmful characters
    if re.search(r"[<>{}]", query):
        raise ValueError("Query contains unsupported characters.")
    return True

# -------------------------------------------
# Async Query Processing
# -------------------------------------------
async def async_answer_query(query_text: str, top_k=DEFAULT_TOP_K):
    """
    Process user query asynchronously with dynamic prompt selection,
    retrieval, and Gemini LLM generation with citations.
    """
    # Validate input
    validate_input(query_text)
    
    # Select appropriate CoT prompt based on question type
    prompt_template = select_prompt_template(query_text)
    
    # Load retriever
    retriever = load_retriever(top_k)
    embeddings_model = get_embeddings()
    
    # Cache query embedding (optional optimization)
    _ = get_query_embedding(query_text, embeddings_model)
    
    # Initialize Gemini LLM
    llm = get_llm()
    
    # Build QA chain with selected prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    try:
        start_time = time.time()
        # Use invoke instead of __call__
        response = qa_chain.invoke({"query": query_text})
        elapsed = time.time() - start_time
        logging.info(f"Query processed in {elapsed:.2f} seconds")
    except Exception as err:
        logging.error(f"Error processing query: {err}")
        return f"Error: {err}", []
    
    # Extract answer and sources
    answer = response.get("result", "No answer generated.")
    documents = response.get("source_documents", [])
    
    # Format citations
    citations = [
        f"[{i + 1}] {doc.metadata.get('source', f'SEC Filing chunk {i + 1}')}"
        for i, doc in enumerate(documents)
    ]
    
    return answer, citations

# -------------------------------------------
# Synchronous Wrapper
# -------------------------------------------
def answer_query(query_text: str, top_k=DEFAULT_TOP_K):
    """Synchronous wrapper for async query processing."""
    return asyncio.run(async_answer_query(query_text, top_k))

# -------------------------------------------
# CLI Interface
# -------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SEC Filings RAG Query Interface")
    print("Powered by Google Gemini 1.5 Flash")
    print("=" * 60)
    print("\nAvailable analysis types:")
    print("  • Risk Analysis")
    print("  • Financial Performance")
    print("  • Liquidity & Solvency")
    print("  • Business Operations")
    print("  • Regulatory & Legal")
    print("  • General SEC Analysis")
    print("\nType 'exit' to quit.\n")
    
    while True:
        try:
            print("-" * 60)
            user_question = input("\nEnter your SEC financial question: ").strip()
            
            if user_question.lower() in {"exit", "quit", "q"}:
                print("\nThank you for using SEC RAG Query Interface. Goodbye!")
                break
            
            # Get top-k parameter
            top_k_input = input(f"Number of chunks to retrieve (default {DEFAULT_TOP_K}): ").strip()
            try:
                top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
                if top_k <= 0 or top_k > 20:
                    print(f"Invalid top-k value. Using default: {DEFAULT_TOP_K}")
                    top_k = DEFAULT_TOP_K
            except ValueError:
                top_k = DEFAULT_TOP_K
            
            # Process query
            print("\nProcessing your query...\n")
            answer, citations = answer_query(user_question, top_k=top_k)
            
            # Display results
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(answer)
            print("\n" + "=" * 60)
            print("CITATIONS:")
            print("=" * 60)
            for citation in citations:
                print(citation)
            print("\n")
            
        except ValueError as ve:
            print(f"\n⚠️  Input Error: {ve}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logging.error(f"Unexpected error: {e}", exc_info=True)
