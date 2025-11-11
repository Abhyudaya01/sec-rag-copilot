import os
import asyncio
import logging
import time
import re
import functools
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import ConfigDict
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


# Load environment variables from .env file
load_dotenv()


# --- Configurations ---
FAISS_INDEX_PATH = "indexes/sec_filings_faiss"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TOP_K = 5
MAX_RETRIES = 3
RETRY_MIN_WAIT = 4
RETRY_MAX_WAIT = 10


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------------------------
# Custom Gemini LLM Wrapper with Enhanced Features
# -------------------------------------------
class GeminiLLM(LLM):
    """
    Custom Gemini LLM wrapper for LangChain with:
    - Automatic retry logic
    - Generation configuration
    - Safety settings
    - Enhanced error handling
    """
    
    model_name: str = GEMINI_MODEL_NAME
    temperature: float = 0
    model: Any = None
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    
    # ✅ FIXED: Pydantic V2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file. Please add:\n"
                "GOOGLE_API_KEY=your-api-key-here"
            )
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Generation configuration for better responses
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_output_tokens=self.max_output_tokens,
        )
        
        # ✅ FIXED: Safety settings - less restrictive to prevent false positives
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Initialize model with configurations
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info(f"✅ Initialized Gemini model: {self.model_name}")
        logger.info(f"   Temperature: {self.temperature}, Max tokens: {self.max_output_tokens}")
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    # ✅ FIXED: Enhanced _call method with proper safety block handling
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry attempt {retry_state.attempt_number}/{MAX_RETRIES}"
        )
    )
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Call Gemini API with automatic retry logic and safety handling.
        Handles finish_reason codes:
        1 = STOP (normal completion)
        2 = SAFETY (blocked by safety filters)
        3 = RECITATION (blocked due to content recitation)
        4 = OTHER (stopped for other reasons)
        """
        try:
            response = self.model.generate_content(prompt)
            
            # CRITICAL: Check candidates exist BEFORE accessing .text
            if not response.candidates:
                logger.warning("⚠️ Response blocked - no candidates returned")
                return "⚠️ Response was blocked by safety filters. Please rephrase your question to be more specific."
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            # Check finish_reason BEFORE accessing .text
            if finish_reason == 2:  # SAFETY
                logger.warning("⚠️ Response blocked due to safety filters (finish_reason=2)")
                return "⚠️ This query was blocked by content safety filters. Try asking a more specific, factual question about financial data."
            
            if finish_reason == 3:  # RECITATION
                logger.warning("⚠️ Response blocked due to recitation (finish_reason=3)")
                return "⚠️ Response blocked due to content policy. Please rephrase your question."
            
            if finish_reason == 4:  # OTHER
                logger.warning("⚠️ Response stopped for other reasons (finish_reason=4)")
                return "⚠️ Unable to generate response. Please try a different question."
            
            # NOW safe to access .text (finish_reason should be 1 = STOP)
            if not response.text or not response.text.strip():
                logger.warning("⚠️ Empty response text")
                return "⚠️ No response generated. Please try rephrasing your question."
            
            return response.text
            
        except ValueError as ve:
            # Catch the specific "response.text" ValueError
            if "finish_reason" in str(ve) or "valid Part" in str(ve):
                logger.warning(f"⚠️ Safety block caught in exception: {ve}")
                return "⚠️ Response blocked by safety filters. Try asking about specific financial metrics instead of general questions."
            raise
            
        except Exception as e:
            logger.error(f"❌ Error calling Gemini API: {e}")
            raise
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async wrapper for _call method."""
        return self._call(prompt, stop)


# -------------------------------------------
# Embeddings Initialization with Caching
# -------------------------------------------
@functools.lru_cache(maxsize=1)
def get_embeddings():
    """
    Initialize and cache the embedding model to avoid reloading.
    Uses HuggingFace's all-MiniLM-L6-v2 for fast, local embeddings.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# -------------------------------------------
# Initialize Gemini LLM
# -------------------------------------------
@functools.lru_cache(maxsize=1)
def get_llm():
    """Initialize and cache custom Gemini LLM."""
    return GeminiLLM()


# -------------------------------------------
# Load FAISS Retriever
# -------------------------------------------
def load_retriever(k=DEFAULT_TOP_K):
    """
    Load FAISS index and return a retriever with dynamic top-k.
    
    Args:
        k: Number of document chunks to retrieve (default: 5)
    
    Returns:
        LangChain retriever object
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Please run document_chunker.py first to create the index."
        )
    
    embeddings = get_embeddings()
    
    # Load FAISS index - safe because we created it ourselves
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    logger.info(f"✅ Loaded FAISS index from {FAISS_INDEX_PATH}")
    
    # Return retriever with specified top-k
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
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

# MD&A Analysis CoT Prompt
MDA_ANALYSIS_COT = """You are an expert in Management Discussion & Analysis (MD&A) interpretation. Using the SEC filing excerpts, analyze management's perspective with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Key trends identification: Review Item 7 MD&A section to identify trends in revenue, expenses, and operations that management highlights.
2. Management's explanation: Extract management's narrative explaining material changes in financial results year-over-year.
3. Forward-looking statements: Identify any forward-looking statements about future operations, capital expenditures, or business outlook.
4. Critical accounting estimates: Note any changes in accounting policies, estimates, or judgments that could affect financial results.
5. Liquidity and capital resources: Assess management's discussion of cash flow adequacy, capital requirements, and funding sources.
6. Cite sources: Reference specific chunks for all management commentary.

Final answer: [Provide comprehensive MD&A analysis with management insights and citations]
"""

# Cash Flow & Capital Allocation CoT Prompt
CASH_FLOW_ALLOCATION_COT = """You are an expert in cash flow analysis and capital allocation strategy. Using the SEC filing excerpts, analyze cash generation and deployment with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Operating cash flow analysis: Review cash from operations, trends, and conversion of earnings to cash.
2. Investing activities: Identify capital expenditures, acquisitions, divestitures, and investment in growth vs. maintenance.
3. Financing activities: Analyze debt issuance/repayment, dividends, share buybacks, and equity transactions.
4. Free cash flow calculation: Compute free cash flow (Operating CF - CapEx) and assess sustainability.
5. Capital allocation priorities: Determine management's capital allocation strategy (growth, returns to shareholders, debt reduction).
6. Cite sources: Reference specific chunks from cash flow statement and MD&A.

Final answer: [Provide cash flow analysis with capital allocation assessment and citations]
"""

# Comparative & Segment Analysis CoT Prompt
COMPARATIVE_SEGMENT_COT = """You are an expert in comparative financial analysis and segment reporting. Using the SEC filing excerpts, compare periods and segments with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Period comparison: Compare current period to prior year and prior quarter (for 10-Q) across key metrics.
2. Percentage changes: Calculate year-over-year and sequential growth rates for revenue, margins, and profitability.
3. Segment breakdown: If applicable, analyze performance by business segment, product line, or geography.
4. Segment profitability: Compare margins, growth rates, and contribution of each segment to total results.
5. Trends and patterns: Identify accelerating/decelerating trends, seasonality, and structural changes.
6. Cite sources: Reference specific chunks for financial data and segment disclosures.

Final answer: [Provide comparative analysis with segment insights and citations]
"""

# Governance & Compensation CoT Prompt
GOVERNANCE_COMPENSATION_COT = """You are an expert in corporate governance and executive compensation analysis. Using the SEC filing excerpts, evaluate governance structures and pay practices with step-by-step reasoning.

Context:
{context}

Question:
{question}

Step-by-step reasoning:
1. Board composition: Review Item 10 for board structure, independence, diversity, and key committee composition (audit, compensation, nominating).
2. Executive compensation: Analyze Item 11 for CEO and top executive pay structure (salary, bonus, equity, benefits).
3. Pay-for-performance alignment: Assess whether executive compensation aligns with company performance and shareholder returns.
4. Insider ownership: Review Item 12 for significant shareholdings by directors, officers, and major investors.
5. Governance red flags: Identify any concerns like related-party transactions (Item 13), lack of independence, or misaligned incentives.
6. Cite sources: Reference specific chunks for governance and compensation data.

Final answer: [Provide governance assessment with compensation analysis and citations]
"""
# SIMPLIFIED PROMPTS (Less likely to trigger safety)
SIMPLE_FINANCIAL_COT = """You are a financial analyst. Using the SEC filing excerpts below, answer the question concisely.

Context:
{context}

Question:
{question}

Provide a clear answer with source references.
"""
# ============================================================================
# FUTURE PREDICTION CoT PROMPT (Structured Reasoning)
# ============================================================================

FUTURE_PREDICTION_COT = """You are a financial analyst performing predictive analysis using historical SEC filing data.

**Your Task:** Make a data-driven prediction about future performance using Chain-of-Thought reasoning.

**Context from SEC Filings:**
{context}

**User Question:**
{question}

**Chain-of-Thought Analysis Process:**

**Step 1: Historical Data Extraction**
- Extract relevant historical metrics (revenue, growth rates, margins, etc.)
- Identify trends over the past 3-5 years
- Note any significant changes or inflection points

**Step 2: Current Performance Assessment**
- Analyze most recent fiscal year performance
- Compare to industry benchmarks
- Identify strengths and weaknesses

**Step 3: Management Outlook & Strategy**
- What does management say about future plans?
- What strategic initiatives are underway?
- What investments is the company making?

**Step 4: Risk Factors Analysis**
- What risks could impact future performance?
- What external factors (market, economy, competition)?
- What internal challenges exist?

**Step 5: Trend Projection**
- Calculate historical growth rates (CAGR)
- Apply reasonable assumptions based on data
- Consider best-case, base-case, worst-case scenarios

**Step 6: Prediction with Confidence Intervals**
- Make a quantitative prediction with ranges
- State assumptions clearly
- Provide confidence level (low/medium/high)

**Final Answer:**
[Provide your prediction with the following structure]

**Prediction:** [Your quantitative forecast]

**Reasoning:**
- Historical trend: [Data-driven observation]
- Management outlook: [From filings]
- Key assumptions: [List 3-5 key assumptions]
- Confidence level: [Low/Medium/High] because [reason]

**Important Caveats:**
- This is a data-driven projection, not financial advice
- Actual results may vary significantly
- Based solely on historical SEC filing data
"""




# -------------------------------------------
# Enhanced Prompt Selection Logic
# -------------------------------------------
def select_prompt_template(question: str) -> PromptTemplate:
    """
    Select the most appropriate CoT prompt based on question keywords.
    Falls back to general SEC analysis prompt if no specific match.
    
    Args:
        question: User's question text
    
    Returns:
        PromptTemplate object with appropriate CoT structure
    """
    question_lower = question.lower()
    
    # MD&A Analysis (Item 7)
    if any(word in question_lower for word in ['mda', 'md&a', 'management discussion', 'management\'s discussion', 'outlook', 'forward-looking', 'guidance', 'future operations', 'critical accounting']):
        template = MDA_ANALYSIS_COT
        logger.info("📊 Selected: MD&A Analysis CoT Prompt (Item 7)")
    
    # Cash Flow & Capital Allocation
    elif any(word in question_lower for word in ['cash flow', 'free cash flow', 'fcf', 'capital allocation', 'capex', 'capital expenditure', 'buyback', 'share repurchase', 'dividend', 'investing activities', 'financing activities']):
        template = CASH_FLOW_ALLOCATION_COT
        logger.info("📊 Selected: Cash Flow & Capital Allocation CoT Prompt")
    
    # Comparative & Segment Analysis
    elif any(word in question_lower for word in ['segment', 'geography', 'geographic', 'product line', 'compare', 'comparison', 'year-over-year', 'yoy', 'y-o-y', 'quarter', 'quarterly', 'sequential', 'trend', 'period']):
        template = COMPARATIVE_SEGMENT_COT
        logger.info("📊 Selected: Comparative & Segment Analysis CoT Prompt")
    
    # Governance & Compensation (Items 10-13)
    elif any(word in question_lower for word in ['governance', 'board', 'director', 'executive compensation', 'ceo pay', 'compensation committee', 'insider', 'ownership', 'beneficial owner', 'related party', 'independence']):
        template = GOVERNANCE_COMPENSATION_COT
        logger.info("📊 Selected: Governance & Compensation CoT Prompt (Items 10-13)")
    
    # Risk Analysis (Item 1A)
    elif any(word in question_lower for word in ['risk', 'risks', 'threat', 'challenge', 'exposure', 'vulnerability', 'uncertainty']):
        template = RISK_ANALYSIS_COT
        logger.info("📊 Selected: Risk Analysis CoT Prompt (Item 1A)")
    
    # Financial Performance (Items 6, 8)
    elif any(word in question_lower for word in ['revenue', 'profit', 'earnings', 'income', 'performance', 'sales', 'margin', 'growth', 'eps', 'earnings per share', 'profitability']):
        template = FINANCIAL_PERFORMANCE_COT
        logger.info("📊 Selected: Financial Performance CoT Prompt (Items 6, 8)")
    
    # Liquidity & Solvency (Balance Sheet)
    elif any(word in question_lower for word in ['liquidity', 'cash', 'debt', 'solvency', 'assets', 'liabilities', 'balance sheet', 'working capital', 'current ratio', 'quick ratio', 'leverage']):
        template = LIQUIDITY_SOLVENCY_COT
        logger.info("📊 Selected: Liquidity & Solvency CoT Prompt")
    
    # Business Operations (Item 1)
    elif any(word in question_lower for word in ['business', 'operations', 'products', 'services', 'market', 'industry', 'competitive', 'competition', 'customer', 'supplier']):
        template = BUSINESS_OPERATIONS_COT
        logger.info("📊 Selected: Business Operations CoT Prompt (Item 1)")
    
    # Regulatory & Legal (Item 3)
    elif any(word in question_lower for word in ['legal', 'lawsuit', 'litigation', 'regulatory', 'compliance', 'violation', 'proceeding', 'investigation']):
        template = REGULATORY_LEGAL_COT
        logger.info("📊 Selected: Regulatory & Legal CoT Prompt (Item 3)")

    # FUTURE PREDICTION ANALYSIS (NEW)
    elif any(word in question_lower for word in ['predict', 'forecast', 'will be', 'next year', 'next 5 years', 'next 10 years', 'future', 'projection']):
        template = FUTURE_PREDICTION_COT
        logger.info("🔮 Selected: Future Prediction CoT Prompt (Data-Driven)")

    # USE SIMPLE PROMPT AS DEFAULT (fallback)
    else:
        template = SIMPLE_FINANCIAL_COT  # Changed from GENERAL_SEC_COT
        logger.info("📊 Selected: Simple Financial Analysis Prompt")
    
    # General SEC Analysis (fallback)
    #else:
        #template = GENERAL_SEC_COT
        #logger.info("📊 Selected: General SEC Analysis CoT Prompt")
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


# -------------------------------------------
# Query Embedding Cache
# -------------------------------------------
query_embedding_cache = {}


def get_query_embedding(query, embeddings_model):
    """
    Cache query embeddings to speed up repeated questions.
    
    Args:
        query: Query text
        embeddings_model: HuggingFace embeddings model
    
    Returns:
        Embedding vector
    """
    if query in query_embedding_cache:
        logger.info("⚡ Using cached query embedding")
        return query_embedding_cache[query]
    
    embedding = embeddings_model.embed_query(query)
    query_embedding_cache[query] = embedding
    return embedding


# -------------------------------------------
# Input Validation
# -------------------------------------------
def validate_input(query):
    """
    Validate and sanitize user input to prevent errors.
    
    Args:
        query: User query string
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If query is invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if len(query) < 3:
        raise ValueError("Query is too short. Please provide more detail.")
    if len(query) > 1000:
        raise ValueError("Query is too long. Please keep it under 1000 characters.")
    
    # Basic sanitization - remove potentially harmful characters
    if re.search(r"[<>{}]", query):
        raise ValueError("Query contains unsupported characters.")
    
    return True


# -------------------------------------------
# Async Query Processing
# -------------------------------------------
async def async_answer_query(query_text: str, top_k=DEFAULT_TOP_K):
    """
    Process user query asynchronously with:
    - Dynamic prompt selection
    - FAISS retrieval
    - Gemini LLM generation
    - Citation extraction
    
    Args:
        query_text: User's question
        top_k: Number of document chunks to retrieve
    
    Returns:
        Tuple of (answer_text, citations_list)
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
        
        # Execute query with invoke method
        response = qa_chain.invoke({"query": query_text})
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Query processed in {elapsed:.2f} seconds")
        
    except Exception as err:
        logger.error(f"❌ Error processing query: {err}", exc_info=True)
        return f"Error processing query: {err}", []
    
    # Extract answer and sources
    answer = response.get("result", "No answer generated.")
    documents = response.get("source_documents", [])
    
    # Format citations with metadata
    citations = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get('source', f'SEC Filing chunk {i + 1}')
        # Add snippet of text for context
        snippet = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        citations.append(f"[{i + 1}] {source}\n    → {snippet}")
    
    return answer, citations

    def normalize_question(question: str) -> str:
        """
        Normalize question to avoid safety filter triggers.
        Expands contractions and standardizes phrasing.
        """
        import re
        
        # Expand contractions
        contractions = {
            "what's": "what is",
            "what'll": "what will",
            "how's": "how is",
            "where's": "where is",
            "apple's": "apple",
            "company's": "company",
            "'s revenue": " revenue",
            "'s profit": " profit",
            "'s growth": " growth"
        }
        
        normalized = question.lower()
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Remove possessive apostrophes that might trigger filters
        normalized = re.sub(r"(\w+)'s\s", r"\1 ", normalized)
        
        logger.info(f"📝 Normalized: '{question}' → '{normalized}'")
        
        return normalized

# -------------------------------------------
# Synchronous Wrapper
# -------------------------------------------
def answer_query(query_text: str, top_k=DEFAULT_TOP_K):
    """
    Synchronous wrapper for async query processing.
    
    Args:
        query_text: User's question
        top_k: Number of document chunks to retrieve
    
    Returns:
        Tuple of (answer_text, citations_list)
    """
    return asyncio.run(async_answer_query(query_text, top_k))


# -------------------------------------------
# CLI Interface
# -------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("🔍 SEC Filings RAG Query Interface")
    print("   Powered by Google Gemini 2.5 Flash + LangChain + FAISS")
    print("=" * 80)
    
    print("\n✨ Available Analysis Types (Auto-Selected Based on Your Question):\n")
    print("  📋 General SEC Analysis")
    print("     └─ Comprehensive analysis of any SEC filing section\n")
    print("  🎯 Risk Analysis (Item 1A)")
    print("     └─ Risk factors, threats, vulnerabilities, exposures\n")
    print("  📈 Financial Performance (Items 6, 8)")
    print("     └─ Revenue, earnings, profitability, growth metrics\n")
    print("  💰 Liquidity & Solvency")
    print("     └─ Cash position, debt, ratios, balance sheet strength\n")
    print("  🏢 Business Operations (Item 1)")
    print("     └─ Products, services, markets, competitive position\n")
    print("  ⚖️  Regulatory & Legal (Item 3)")
    print("     └─ Legal proceedings, compliance, investigations\n")
    print("  📊 MD&A Analysis (Item 7)")
    print("     └─ Management discussion, outlook, forward-looking statements\n")
    print("  💵 Cash Flow & Capital Allocation")
    print("     └─ Cash generation, CapEx, dividends, buybacks\n")
    print("  📉 Comparative & Segment Analysis")
    print("     └─ Period comparisons, segment breakdowns, trends\n")
    print("  👔 Governance & Compensation (Items 10-13)")
    print("     └─ Board structure, executive pay, insider ownership\n")
    
    print("=" * 80)
    print("\n💡 Tips for Best Results:")
    print("  • Be specific: 'What were Apple's top 3 risks in 2024?' vs. 'Tell me about risks'")
    print("  • Use keywords: Include terms like 'segment', 'cash flow', 'MD&A' for better routing")
    print("  • Ask follow-ups: Each query learns from context for more refined answers")
    print("  • Type 'exit', 'quit', or 'q' to end session\n")
    
    print("📌 Example Questions:")
    print("  • What are the main risk factors for Apple?")
    print("  • How much did revenue grow year-over-year?")
    print("  • What does management say about future outlook in MD&A?")
    print("  • Analyze Apple's iPhone segment performance")
    print("  • What is the company's free cash flow and capital allocation strategy?\n")
    
    print("=" * 80)
    
    while True:
        try:
            print("-" * 80)
            user_question = input("\n❓ Enter your SEC financial question: ").strip()
            
            if user_question.lower() in {"exit", "quit", "q"}:
                print("\n👋 Thank you for using SEC RAG Query Interface. Goodbye!")
                break
            
            if not user_question:
                print("⚠️  Please enter a valid question.")
                continue
            
            # Get top-k parameter
            top_k_input = input(f"📚 Number of chunks to retrieve (default {DEFAULT_TOP_K}, press Enter to skip): ").strip()
            try:
                top_k = int(top_k_input) if top_k_input else DEFAULT_TOP_K
                if top_k <= 0 or top_k > 20:
                    print(f"⚠️  Invalid top-k value. Using default: {DEFAULT_TOP_K}")
                    top_k = DEFAULT_TOP_K
            except ValueError:
                top_k = DEFAULT_TOP_K
            
            # Process query
            print("\n⏳ Processing your query with Chain-of-Thought reasoning...\n")
            answer, citations = answer_query(user_question, top_k=top_k)
            
            # Display results
            print("\n" + "=" * 80)
            print("📋 ANSWER:")
            print("=" * 80)
            print(answer)
            
            if citations:
                print("\n" + "=" * 80)
                print("📎 CITATIONS & SOURCE EXCERPTS:")
                print("=" * 80)
                for citation in citations:
                    print(citation)
                    print()
            
            print("\n" + "-" * 80)
            print()
            
        except ValueError as ve:
            print(f"\n⚠️  Input Error: {ve}\n")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logger.error(f"Unexpected error: {e}", exc_info=True)
