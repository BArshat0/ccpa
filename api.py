import os
import sys
import json
import re
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app import get_vector_db, get_llm, CCPAComplianceCheck
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Global variables to hold model state
retrieval_chain = None
is_ready = False
startup_error: str | None = None

SECTION_ID_PATTERN = re.compile(
    r"1798\.\d+(?:\.\d+)?(?:\([a-z0-9]+\))*",
    re.IGNORECASE
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
    "their", "they", "them", "our", "are", "was", "were", "have", "has", "had",
    "will", "would", "should", "could", "not", "without", "about", "what", "when",
    "where", "which", "while", "all", "any", "can", "may", "but", "its", "being",
    "been", "than", "then", "too", "out", "use", "using", "over", "under", "after",
    "before", "between", "within", "through", "across", "such", "per"
}


def _normalize_section(raw_text: str) -> str | None:
    if not raw_text:
        return None
    match = SECTION_ID_PATTERN.search(raw_text.lower())
    if not match:
        return None
    normalized = match.group(0)
    return f"Section {normalized}"


def _extract_sections(text: str) -> list[str]:
    if not text:
        return []
    seen = set()
    sections = []
    for match in SECTION_ID_PATTERN.finditer(text.lower()):
        section = f"Section {match.group(0)}"
        if section not in seen:
            seen.add(section)
            sections.append(section)
    return sections


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for token in TOKEN_PATTERN.findall((text or "").lower()):
        if len(token) < 3 or token in STOPWORDS or token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _score_chunk_relevance(prompt_tokens: set[str], chunk_text: str) -> int:
    if not prompt_tokens or not chunk_text:
        return 0
    chunk_tokens = _tokenize(chunk_text)
    if not chunk_tokens:
        return 0
    return len(prompt_tokens.intersection(chunk_tokens))


def _collect_grounded_sections(context_docs, prompt: str, max_sections: int = 5) -> list[str]:
    prompt_tokens = _tokenize(prompt)
    section_scores: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for idx, doc in enumerate(context_docs or []):
        text = (getattr(doc, "page_content", "") or "").strip()
        lowered = text.lower()
        if not text:
            continue
        # Ignore table-of-contents style chunks that list many sections and degrade precision.
        if "table of contents" in lowered:
            continue

        sections = _extract_sections(text)
        if not sections or len(sections) > 8:
            continue

        score = _score_chunk_relevance(prompt_tokens, text)
        for section in sections:
            if section not in first_seen:
                first_seen[section] = idx
            section_scores[section] = max(section_scores.get(section, 0), score)

    ranked = sorted(
        section_scores.items(),
        key=lambda item: (-item[1], first_seen[item[0]])
    )
    return [section for section, _ in ranked[:max_sections]]


def _infer_sections_from_prompt(prompt: str) -> list[str]:
    prompt_lc = (prompt or "").lower()
    inferred = []

    if (
        "delete" in prompt_lc
        and any(term in prompt_lc for term in ["request", "consumer", "customer"])
    ):
        inferred.append("Section 1798.105")
    if any(term in prompt_lc for term in ["sell", "selling", "data broker", "opt out", "do not sell"]):
        inferred.append("Section 1798.120")
    if any(term in prompt_lc for term in ["collect", "collection", "privacy policy", "notice"]):
        inferred.append("Section 1798.100")
    if any(term in prompt_lc for term in ["discriminat", "higher price", "different price", "penalty"]):
        inferred.append("Section 1798.125")

    # Preserve order and deduplicate
    seen = set()
    unique = []
    for section in inferred:
        if section not in seen:
            seen.add(section)
            unique.append(section)
    return unique


def _resolve_articles(prompt: str, model_articles: list[str], grounded_sections: list[str]) -> list[str]:
    normalized_model = []
    seen_model = set()
    for item in model_articles or []:
        for section in _extract_sections(item):
            if section not in seen_model:
                seen_model.add(section)
                normalized_model.append(section)
        normalized_single = _normalize_section(item)
        if normalized_single and normalized_single not in seen_model:
            seen_model.add(normalized_single)
            normalized_model.append(normalized_single)

    grounded_set = set(grounded_sections)
    if grounded_sections:
        trusted_model = [s for s in normalized_model if s in grounded_set]
        if trusted_model:
            return trusted_model

        inferred = _infer_sections_from_prompt(prompt)
        trusted_inferred = [s for s in inferred if s in grounded_set]
        if trusted_inferred:
            return trusted_inferred

        return grounded_sections[:3]

    if normalized_model:
        return normalized_model[:3]

    return _infer_sections_from_prompt(prompt)[:3]

def _initialize_models():
    global retrieval_chain, is_ready, startup_error
    print("Starting up and loading models... This might take a few minutes.")

    try:
        # Load Chroma DB and Retriever
        vector_db = get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        
        # Load the LLM (Qwen2.5-1.5B)
        llm = get_llm()
        
        # Initialize parser explicitly to ensure json formatting
        parser = PydanticOutputParser(pydantic_object=CCPAComplianceCheck)
        
        # Prompt template tailored to Qwen ChatML format
        template = """<|im_start|>system
You are a helpful assistant analyzing actions for CCPA compliance. Determine if the action described in the user input is a CCPA violation ("harmful") and list the relevant CCPA articles. If the action is legally compliant, follows the rules, or is unrelated to CCPA, it is NOT harmful (set harmful=false).
You MUST cite only sections that explicitly appear in the provided Context. Do not invent or guess citations.
Return citations in canonical format: "Section 1798.xxx".

CRITICAL: Output ONLY a single valid JSON object. Do not explain, do not add introductory text, do not converse. Do NOT wrap the JSON in markdown formatting or code blocks.
{format_instructions}<|im_end|>
<|im_start|>user
Context: {context}

Question/Scenario: {input}<|im_end|>
<|im_start|>assistant
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # Mark health check as ready
        is_ready = True
        startup_error = None
        print("Models loaded successfully. API is ready.")
    except Exception as e:
        startup_error = str(e)
        print(f"Failed to load models during startup: {e}")

# Create the startup event using lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_initialize_models, daemon=True).start()
    yield  # API serves requests here
    
    print("Shutting down API...")

app = FastAPI(lifespan=lifespan)
parser = PydanticOutputParser(pydantic_object=CCPAComplianceCheck)

# Request Models
class AnalyzeRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health_check():
    """
    Health check endpoint: will return 200 OK only when models are fully loaded in memory.
    """
    if is_ready:
        return {"status": "ok"}
    if startup_error:
        return JSONResponse(status_code=500, content={"status": "error", "detail": startup_error})
    return JSONResponse(status_code=503, content={"status": "loading"})

@app.post("/analyze")
async def analyze_prompt(request: AnalyzeRequest):
    """
    Accepts a prompt, passes to RAG LLM pipeline, and performs logic formatting checks on the JSON parsing.
    Saves outputs to a log file.
    """
    global retrieval_chain
    if startup_error:
        return JSONResponse(
            status_code=500,
            content={"error": f"Startup failed: {startup_error}"}
        )
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not yet loaded. Please try again later."}
        )
        
    print(f"\nAnalyzing prompt: '{request.prompt}'")
    
    # 1. Invoke the LangChain retrieval chain
    try:
        response = retrieval_chain.invoke({"input": request.prompt})
        raw_answer = response.get("answer", "").strip()
        grounded_sections = _collect_grounded_sections(response.get("context", []), request.prompt)
        
        # 2. Parse into Pydantic model
        try:
            parsed_output = parser.parse(raw_answer)
        except Exception as e:
            print(f"Pydantic parsing failed: {e}. Attempting JSON extraction fallback...")
            json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
            if json_match:
                extracted = json_match.group(0)
                try:
                    parsed_data = json.loads(extracted)
                    parsed_output = CCPAComplianceCheck(**parsed_data)
                except Exception as ex:
                    raise Exception(f"Fallback extraction failed: {ex}")
            else:
                raise Exception("No JSON object found in output.")
        
        # 3. Logic validation
        if parsed_output.harmful:
            parsed_output.articles = _resolve_articles(
                prompt=request.prompt,
                model_articles=parsed_output.articles,
                grounded_sections=grounded_sections
            )
        else:
            # If harmful is false, articles must be empty
            if parsed_output.articles:
                print("Warning: harmful is false but articles were returned. Clearing articles list.")
                parsed_output.articles = []
                
        # Format the validated response into a dict
        final_response = {
            "harmful": parsed_output.harmful,
            "articles": parsed_output.articles
        }
    except Exception as e:
        print(f"Failed to compile acceptable JSON. Error: {e}")
        # Build an emergency fallback in case the LLM breaks syntax completely
        final_response = {
            "harmful": False,
            "articles": []
        }
        
    # 5. Logging Results
    # Append the request/response to a local log file for organisms to review
    with open("api_results_log.jsonl", "a") as f:
        log_entry = {
            "prompt": request.prompt,
            "response": final_response
        }
        f.write(json.dumps(log_entry) + "\n")
        
    return final_response

if __name__ == "__main__":
    import uvicorn
    # Launch uvicorn locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)