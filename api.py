import os
import sys
import json
import re
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader
from app import get_vector_db, get_llm, CCPAComplianceCheck
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Global variables to hold model state
retrieval_chain = None
is_ready = False
startup_error: str | None = None
statute_section_corpus: dict[str, dict[str, object]] = {}
prompt_refiner_llm = None

SECTION_ID_PATTERN = re.compile(
    r"1798\.\d+(?:\.\d+)?(?:\([a-z0-9]+\))*",
    re.IGNORECASE
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STATUTE_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(1798\.\d+(?:\.\d+)?(?:\([a-z0-9]+\))?)\.\s+(.*)$",
    re.IGNORECASE
)
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "you",
    "their", "they", "them", "our", "are", "was", "were", "have", "has", "had",
    "will", "would", "should", "could", "not", "without", "about", "what", "when",
    "where", "which", "while", "all", "any", "can", "may", "but", "its", "being",
    "been", "than", "then", "too", "out", "use", "using", "over", "under", "after",
    "before", "between", "within", "through", "across", "such", "per"
}

LEGAL_GENERIC_TOKENS = {
    "consumer", "consumers", "business", "businesses", "personal", "information",
    "right", "rights", "request", "requests", "shall", "section", "title",
    "collects", "collected", "collection", "processing", "data"
}

COMPLIANT_PATTERNS = [
    "clear privacy policy",
    "allows customers to opt out",
    "allow customers to opt out",
    "do not sell my personal information link",
    "as required",
    "deleted all personal data within 45 days",
    "within 45 days",
    "equal service and pricing",
    "regardless of whether they exercise",
    "honor all deletion requests",
]

VIOLATION_CUES = [
    "without",
    "not",
    "no ",
    "ignore",
    "ignoring",
    "refuse",
    "refused",
    "denied",
    "higher price",
    "different price",
    "penal",
    "fail",
    "failed",
]


def _canonical_section(section_id: str) -> str:
    return f"Section {section_id}"


def _build_statute_section_corpus(pdf_path: str = "ccpa_statute.pdf") -> dict[str, dict[str, object]]:
    if not os.path.exists(pdf_path):
        return {}

    reader = PdfReader(pdf_path)
    # Skip cover/contents pages to avoid TOC contamination.
    statute_text = "\n".join((page.extract_text() or "") for page in reader.pages[2:])
    statute_text = statute_text.replace("\r", "\n")

    matches = list(STATUTE_HEADING_PATTERN.finditer(statute_text))
    if not matches:
        return {}

    corpus: dict[str, dict[str, object]] = {}
    for idx, match in enumerate(matches):
        section_id = match.group(1).lower()
        title = " ".join(match.group(2).split())
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(statute_text)
        body = " ".join(statute_text[start:end].split())

        # Keep first full-body occurrence only.
        if section_id in corpus:
            continue

        body_tokens = _tokenize(body)
        title_tokens = _tokenize(title)
        corpus[section_id] = {
            "title": title,
            "body": body,
            "title_tokens": title_tokens,
            "body_tokens": body_tokens,
        }

    return corpus


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


def _collect_grounded_section_scores(context_docs, prompt: str, max_sections: int = 6) -> dict[str, int]:
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
    return {section: score for section, score in ranked[:max_sections]}


def _statute_phrase_boosts(prompt_lc: str, section_id: str) -> int:
    boost = 0

    # 1798.100 - collection notice and disclosure duties
    if section_id == "1798.100":
        if any(p in prompt_lc for p in ["collect", "collection", "privacy policy", "notice at collection"]):
            boost += 4
        if any(p in prompt_lc for p in ["without informing", "without notifying", "doesn't mention", "not mention"]):
            boost += 3

    # 1798.105 - deletion
    if section_id == "1798.105":
        if any(p in prompt_lc for p in ["delete", "deletion", "erase", "remove data"]):
            boost += 5
        if any(p in prompt_lc for p in ["ignore request", "refuse", "denied request", "keeping all records"]):
            boost += 3

    # 1798.106 - correction
    if section_id == "1798.106":
        if any(p in prompt_lc for p in ["correct inaccurate", "correction request", "inaccurate personal information"]):
            boost += 5

    # 1798.110 - right to know/access collected PI
    if section_id == "1798.110":
        if any(p in prompt_lc for p in ["right to know", "access request", "what personal information", "categories collected"]):
            boost += 5

    # 1798.115 - right to know sold/shared PI and recipients
    if section_id == "1798.115":
        if any(p in prompt_lc for p in ["shared with", "sold to", "third party recipients", "to whom"]):
            boost += 4

    # 1798.120 - opt-out of sale/sharing; minors consent for sale/sharing
    if section_id == "1798.120":
        if any(p in prompt_lc for p in ["sell", "selling", "data broker", "opt out", "do not sell", "share personal information"]):
            boost += 6
        if any(p in prompt_lc for p in ["minor", "under 16", "13-year", "14-year", "without consent", "parent consent"]):
            boost += 4

    # 1798.121 - sensitive PI limitation
    if section_id == "1798.121":
        if any(p in prompt_lc for p in ["sensitive personal information", "limit use", "limit disclosure"]):
            boost += 5

    # 1798.125 - non-discrimination / retaliation
    if section_id == "1798.125":
        if any(p in prompt_lc for p in ["discriminat", "higher price", "different price", "deny service", "retaliat"]):
            boost += 6

    # 1798.130 / 1798.135 - request-handling and links/notice methods
    if section_id == "1798.130":
        if any(p in prompt_lc for p in ["request method", "toll-free", "response timeline", "verifiable consumer request"]):
            boost += 4
    if section_id == "1798.135":
        if any(p in prompt_lc for p in ["do not sell or share", "limit the use of my sensitive", "homepage link"]):
            boost += 5

    # 1798.150 - security breach private right of action
    if section_id == "1798.150":
        if any(p in prompt_lc for p in ["data breach", "security breach", "unauthorized access", "exfiltration", "stolen"]):
            boost += 5

    return boost


def _section_allowed_by_prompt(section: str, prompt_lc: str) -> bool:
    if section == "Section 1798.125":
        return any(k in prompt_lc for k in ["discriminat", "higher price", "different price", "deny service", "retaliat", "penal"])
    if section == "Section 1798.105":
        return any(k in prompt_lc for k in ["delete", "deletion", "erase", "remove data"])
    if section == "Section 1798.120":
        return any(k in prompt_lc for k in ["sell", "selling", "data broker", "opt out", "do not sell", "share personal information", "minor", "under 16"])
    if section == "Section 1798.100":
        return any(k in prompt_lc for k in ["collect", "collecting", "privacy policy", "notice", "without informing", "without notifying", "does not mention", "doesn't mention"])
    if section == "Section 1798.121":
        return any(k in prompt_lc for k in ["sensitive personal information", "limit use", "limit disclosure"])
    if section == "Section 1798.150":
        return any(k in prompt_lc for k in ["data breach", "security breach", "unauthorized access", "exfiltration", "stolen"])
    return True


def _rule_based_decision(prompt: str) -> tuple[bool | None, list[str]]:
    prompt_lc = (prompt or "").lower()
    sections: list[str] = []

    def add(section_id: str):
        section = _canonical_section(section_id)
        if section not in sections:
            sections.append(section)

    # Explicitly compliant phrasing from common CCPA-compliant scenarios.
    if any(p in prompt_lc for p in COMPLIANT_PATTERNS):
        return False, []

    has_violation_cue = any(cue in prompt_lc for cue in VIOLATION_CUES)

    # 1798.105 deletion request ignored/denied
    if "delete" in prompt_lc and any(k in prompt_lc for k in ["request", "consumer", "customer"]):
        if any(k in prompt_lc for k in ["ignore", "ignoring", "refuse", "refused", "denied", "keeping all records"]):
            add("1798.105")

    # 1798.120 sale/share without opt-out or without required consent
    if any(k in prompt_lc for k in ["sell", "selling", "data broker", "share personal information", "sharing personal information"]):
        if any(k in prompt_lc for k in ["without", "no chance to opt out", "without opt out", "no opt out", "without consent"]):
            add("1798.120")
        if any(k in prompt_lc for k in ["minor", "under 16", "13-year", "14-year", "parent"]):
            add("1798.120")

    # 1798.100 notice at collection / disclosure duties
    if any(k in prompt_lc for k in ["collect", "collecting", "privacy policy", "notice"]):
        if any(k in prompt_lc for k in ["doesn't mention", "not mention", "without informing", "without notifying", "does not mention"]):
            add("1798.100")

    # 1798.125 discriminatory treatment for rights exercise
    if any(k in prompt_lc for k in ["higher price", "different price", "discriminat", "deny service", "penal"]):
        if any(k in prompt_lc for k in ["opt out", "privacy rights", "exercise rights", "do not sell"]):
            add("1798.125")

    # 1798.135 missing required do-not-sell/limit-use mechanisms
    if "do not sell" in prompt_lc and any(k in prompt_lc for k in ["missing", "no link", "not available", "cannot find"]):
        add("1798.135")

    # 1798.150 breach/security harm
    if any(k in prompt_lc for k in ["data breach", "security breach", "unauthorized access", "stolen personal information", "exfiltration"]):
        add("1798.150")

    if sections:
        return True, sections

    # When no specific section is confidently mapped, avoid overriding the model.
    if has_violation_cue:
        return None, []
    return None, []


def _score_sections_from_statute(prompt: str, max_sections: int = 6) -> dict[str, int]:
    if not statute_section_corpus:
        return {}

    prompt_lc = (prompt or "").lower()
    prompt_tokens = _tokenize(prompt)
    prompt_specific = {t for t in prompt_tokens if t not in LEGAL_GENERIC_TOKENS}
    scores: dict[str, int] = {}

    for section_id, meta in statute_section_corpus.items():
        title_tokens = {t for t in meta["title_tokens"] if t not in LEGAL_GENERIC_TOKENS}
        body_tokens = {t for t in meta["body_tokens"] if t not in LEGAL_GENERIC_TOKENS}

        # Emphasize heading semantics first, then body overlap.
        overlap_title = len(prompt_specific.intersection(title_tokens))
        overlap_body = min(2, len(prompt_specific.intersection(body_tokens)))
        score = (overlap_title * 5) + overlap_body + _statute_phrase_boosts(prompt_lc, section_id)
        section = _canonical_section(section_id)
        if not _section_allowed_by_prompt(section, prompt_lc):
            score -= 5
        if score > 0:
            scores[section] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {section: score for section, score in ranked[:max_sections]}


def _resolve_articles(
    prompt: str,
    model_articles: list[str],
    grounded_scores: dict[str, int],
    rule_sections: list[str] | None = None,
) -> list[str]:
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

    statute_scores = _score_sections_from_statute(prompt)
    prompt_lc = (prompt or "").lower()
    grounded_set = set(grounded_scores.keys())
    rule_set = set(rule_sections or [])
    candidates = set(normalized_model) | set(statute_scores.keys()) | grounded_set | rule_set

    ranked_scores: dict[str, int] = {}
    for section in candidates:
        score = 0
        if section in rule_set:
            score += 10
        score += grounded_scores.get(section, 0) * 5
        score += statute_scores.get(section, 0) * 2
        if section in normalized_model:
            score += 2
        if section in grounded_set and section in normalized_model:
            score += 2
        if not _section_allowed_by_prompt(section, prompt_lc):
            score -= 8
        # Keep only citations that exist in the parsed statute corpus when available.
        if statute_section_corpus:
            sec_id = section.replace("Section ", "").lower()
            if sec_id not in statute_section_corpus:
                score = 0
        ranked_scores[section] = score

    ranked = sorted(
        [(s, sc) for s, sc in ranked_scores.items() if sc > 0],
        key=lambda item: item[1],
        reverse=True
    )

    # Prioritize sections that are explicitly retrieved from context.
    grounded_ranked = [section for section, _ in ranked if section in grounded_set]
    if grounded_ranked:
        return grounded_ranked[:3]

    if ranked:
        return [section for section, _ in ranked[:3]]

    return normalized_model[:3]


def _refine_prompt_with_llm(original_prompt: str) -> str:
    if not original_prompt or not prompt_refiner_llm:
        return original_prompt

    refinement_prompt = f"""<|im_start|>system
You rewrite compliance prompts for legal analysis quality.
Rules:
1) Preserve original meaning and facts exactly.
2) Do not add or invent new facts.
3) Make the scenario explicit about action, actor, and potential policy/legal issue.
4) Output one plain sentence only.
<|im_end|>
<|im_start|>user
Original prompt: {original_prompt}
<|im_end|>
<|im_start|>assistant
"""

    try:
        refined = prompt_refiner_llm.invoke(refinement_prompt)
        refined = str(refined).strip()
        refined = re.sub(r"^(refined prompt|rewritten prompt)\s*:\s*", "", refined, flags=re.IGNORECASE)
        refined = re.sub(r"\s+", " ", refined).strip().strip("\"'")

        if not refined:
            return original_prompt
        # Guardrail: if refinement looks malformed, keep original.
        if len(refined) < 12:
            return original_prompt
        return refined[:1200]
    except Exception as e:
        print(f"Prompt refinement failed, using original prompt. Error: {e}")
        return original_prompt


def _initialize_models():
    global retrieval_chain, is_ready, startup_error, statute_section_corpus, prompt_refiner_llm
    print("Starting up and loading models... This might take a few minutes.")

    try:
        # Load Chroma DB and Retriever
        vector_db = get_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 8})
        
        # Load the LLM (Qwen2.5-1.5B)
        llm = get_llm()
        prompt_refiner_llm = llm
        
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

        # Build searchable section corpus from the full statute for citation ranking.
        statute_section_corpus = _build_statute_section_corpus()
        print(f"Loaded statute section corpus with {len(statute_section_corpus)} sections.")
        
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
        refined_prompt = _refine_prompt_with_llm(request.prompt)
        if refined_prompt != request.prompt:
            print(f"Refined prompt: '{refined_prompt}'")

        response = retrieval_chain.invoke({"input": refined_prompt})
        raw_answer = response.get("answer", "").strip()
        grounded_scores = _collect_grounded_section_scores(response.get("context", []), refined_prompt)
        
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
        rule_harmful, rule_sections = _rule_based_decision(request.prompt)

        if rule_harmful is False:
            parsed_output.harmful = False
            parsed_output.articles = []
        elif parsed_output.harmful or rule_harmful is True:
            parsed_output.harmful = True
            parsed_output.articles = _resolve_articles(
                prompt=request.prompt,
                model_articles=parsed_output.articles,
                grounded_scores=grounded_scores,
                rule_sections=rule_sections
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
