import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def main():
    print(f"Downloading LLM model ({MODEL_ID}) to cache...")
    AutoTokenizer.from_pretrained(MODEL_ID)
    AutoModelForCausalLM.from_pretrained(MODEL_ID)

    print(f"Downloading embedding model ({EMBEDDING_MODEL}) to cache...")
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Pre-download complete.")

if __name__ == "__main__":
    main()
