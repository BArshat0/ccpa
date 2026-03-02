import os
import sys
import shutil
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Set environment variables for Hugging Face dynamically
if "HF_TOKEN" in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HF_TOKEN"]
else:
    print("Warning: HF_TOKEN environment variable not set. Models may fail to download if gated.")

DB_DIR = "./chroma_db"
PDF_PATH = "ccpa_statute.pdf"
FORCE_REBUILD_DB = os.getenv("FORCE_REBUILD_DB", "0") == "1"
# Qwen2.5 1.5B Instruct model to fit within 8GB VRAM limit while keeping JSON parsing abilities
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

class CCPAComplianceCheck(BaseModel):
    harmful: bool = Field(description="true if the prompt describes a CCPA violation, false otherwise")
    articles: list[str] = Field(default_factory=list, description="list of strings, e.g. ['Section 1798.100']. Non-empty when harmful=true. Must be [] when harmful=false.")

def get_vector_db():
    print("Loading embedding model...")
    # Use a stronger embedding model 
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # Check if vector DB persist directory already exists
    if os.path.exists(DB_DIR) and os.path.isdir(DB_DIR) and not FORCE_REBUILD_DB:
        print(f"Loading existing vector database from '{DB_DIR}'...")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        if not os.path.exists(PDF_PATH):
            print(f"Error: Could not find '{PDF_PATH}'. Please ensure it is in the current directory.")
            sys.exit(1)

        if FORCE_REBUILD_DB and os.path.exists(DB_DIR) and os.path.isdir(DB_DIR):
            print("FORCE_REBUILD_DB=1 detected. Rebuilding vector database...")
            shutil.rmtree(DB_DIR, ignore_errors=True)
            
        print(f"Creating new vector database from '{PDF_PATH}'...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # Larger chunks preserve section headers with surrounding obligations for better citation grounding.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        docs = text_splitter.split_documents(documents)
        
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print("Vector database created and saved successfully.")
        
    return vector_db

def get_llm():
    print(f"Downloading/Loading LLM: {MODEL_ID}...")
    
    # Check if GPU is available to use 8-bit quantization with bitsandbytes
    if torch.cuda.is_available():
        print("GPU detected. Using 8-bit quantization (bitsandbytes)...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("No GPU detected. Loading model across CPU...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Create the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False # prevents repeating the prompt in the output
    )
    
    # Return as LangChain component
    return HuggingFacePipeline(pipeline=pipe)

def main():
    # 1. Initialize DB and Retriever
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    
    # 2. Initialize LLM Pipeline
    llm = get_llm()
    
    # 3. Create RAG Chain
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
    
    print("\n--- RAG Application Ready ---")
    while True:
        try:
            query = input("\nEnter your question about CCPA (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("Thinking...")
            response = retrieval_chain.invoke({"input": query})
            
            raw_answer = response.get("answer", "").strip()
            print("\nRaw Answer:")
            print(raw_answer)
            
            try:
                parsed_output = parser.parse(raw_answer)
                print("\nStructured JSON Output:")
                print(parsed_output.model_dump_json(indent=2))
            except Exception as e:
                print("\nFailed to parse output into strict JSON format:")
                print(e)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
