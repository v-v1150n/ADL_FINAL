from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# 定義使用的 LLM 模型和嵌入模型名稱
llm_name = "kenneth85/llama-3-taiwan"  
embedding_name = "sentence-transformers/all-MiniLM-L12-v2"

# 初始化 LLM，使用 LangChain 提供的 ChatOllama 接口
llm = ChatOllama(
    model=llm_name,  
    temperature=0.3  
)

# 初始化嵌入模型，使用 LangChain 提供的 HuggingFaceEmbeddings 接口
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_name,  
    model_kwargs={'device': "cuda"},  
    encode_kwargs={'normalize_embeddings': True},  
    show_progress=True  
)
