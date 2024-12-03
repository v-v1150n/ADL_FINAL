from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings  

# llm_name = "willh/taide-lx-7b-chat-4bit"
llm_name = "kenneth85/llama-3-taiwan"
embedding_name = "all-MiniLM-L6-v2"

llm = ChatOllama(model=llm_name, temperature=0.3)

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_name,
    model_kwargs={'device': "cuda"},  
    encode_kwargs={'normalize_embeddings': True}, 
    show_progress=True  
)

