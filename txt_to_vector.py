from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

def load_and_split_documents(file_path, chunk_size, chunk_overlap):
    # 使用 TextLoader 加載文件內容
    loader = TextLoader(file_path)
    documents = loader.load()
    # 使用 CharacterTextSplitter 進行文本分割
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def initialize_embeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu", normalize_embeddings=False):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings},
        show_progress=True   # 顯示進度條
    )

def save_to_chroma(docs, embeddings, output_path):
    try:
        # 將文檔保存到向量數據庫
        Chroma.from_documents(docs, embeddings, persist_directory=output_path)
        print(f"向量數據庫已成功保存到 {output_path}")
    except Exception as e:
        print(f"保存向量數據庫時發生錯誤: {e}")

def main(file_path, chunk_size, chunk_overlap, iu_id, model_name):
    # 設置輸出路徑
    output_path = f'./VECTOR_DB/{iu_id}'
    # 加載並分割文檔
    docs = load_and_split_documents(file_path, chunk_size, chunk_overlap)
    # 初始化嵌入模型
    hf = initialize_embeddings(model_name=model_name)
    # 保存文檔到向量數據庫
    save_to_chroma(docs, hf, output_path)

if __name__ == "__main__":
    # 手動設置參數
    FILE_PATH =  "TO_RAG/Benzene_alternatives_Childrens_Products.txt" # 替換為你的文件路徑
    CHUNK_SIZE = 1500  # 文本塊大小
    CHUNK_OVERLAP = 150  # 文本塊重疊大小
    IU_ID = "CHILDRENS_PRODUCTS"  # 工業用途 ID
    MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
 
    # 執行主函數
    main(FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, IU_ID, MODEL_NAME)

