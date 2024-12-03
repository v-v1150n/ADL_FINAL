import requests
import logging
import json
import streamlit as st
from datetime import datetime
import retriever_chain as rag

logging.basicConfig(level=logging.INFO)

def get_id_from_url():
    chemical_id = st.query_params.id
    if chemical_id:
        return chemical_id  # Retrieve the ID from URL
    else:
        st.warning("No Chemical ID provided in URL")
        return None
    
def save_conversation_to_json(user_input, chatbot_response):
    """
    將對話數據和檢索結果保存為 JSON 文件
    """
    data = {
        "user_input": user_input,
        # "retrieved_documents": [{"content": doc} for doc in retrieved_documents],  # 保存檢索內容
        "chatbot_response": chatbot_response,
        "timestamp": datetime.now().isoformat()  # 添加時間戳
    }
    output_path = "conversation_logs.json"

    try:
        # 檢查文件是否已存在，若存在則讀取後追加
        try:
            with open(output_path, 'r') as file:
                conversations = json.load(file)
        except FileNotFoundError:
            conversations = []

        conversations.append(data)

        # 保存文件
        with open(output_path, 'w') as file:
            json.dump(conversations, file, ensure_ascii=False, indent=4)
        print(f"對話數據已保存到 {output_path}")
    except Exception as e:
        print(f"保存數據時出錯: {e}")

def get_response(query):
    try:
        # 設定檢索的向量資料庫
        load_path = ["./BENZENE_CHROMA_DB"]
        chain = rag.chain(load_path=load_path)

        # # 檢索文檔
        # retrieved_documents = chain.retrieve(query)  # 假設 retriever 支持這個方法
        # logging.info(f"Retrieved Documents: {retrieved_documents}")

        # 調用模型生成回應
        response = chain.invoke(query)
        logging.info(f"Response from chain.invoke(): {response}")

        # 保存對話紀錄
        save_conversation_to_json(
            user_input=query,
            # retrieved_documents=retrieved_documents,
            chatbot_response=response
        )

        return response
    except AttributeError as e:
        logging.error(f"AttributeError: {e}")
        return f"處理請求時出錯: {e}"
    except Exception as e:
        logging.error(f"Exception: {e}")
        return f"處理請求時出錯: {e}"


def main():
    chemical_name = get_api_response("https://sas.cmdm.tw/api/chemicals/name/59")

    st.title('🧪 SAS GPT 對談機器人')
    st.caption("🦙 A SAS GPT powered by Llama-3-Taiwan-8B & NeMo-Guardrails") 
    st.warning(f'🤖 請詢問有關 🧪 {chemical_name}的相關問題，目前對談機器人基於SAS系統整理的危害資訊以及安全替代物回答問題，但仍建議您再次確認。您可嘗試提問：「{chemical_name}有什麼危害資訊」、「{chemical_name}有什麼安全替代物」')

    with st.sidebar:
        st.button('🧹 清除查詢記錄', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "請提問化學物質相關問題"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("請提問化學物質相關問題"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        query = f"關於{chemical_name}，" + prompt
        logging.info(f"提問：{query}")

        with st.spinner("思考中，請稍候..."):
            response = get_response(query)
            logging.info(f"回覆：{response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

def is_summary_query(query):
    summary_keywords = ["總結", "概述", "摘要", "回顧", "重點", "要點", "整理", "summary", "summarize", "summarization", "conclude"]
    return any(keyword in query for keyword in summary_keywords)


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "請輸入化學物質相關問題"}]

def get_api_response(url):
    try:
        response = requests.get(url)
        content_type = response.headers.get('Content-Type')

        if 'application/json' in content_type:
            return response.json()
        elif 'text/plain' in content_type:
            return response.text
        else:
            logging.debug(f"Unhandled Content-Type: {content_type}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API請求出錯: {e}")
        return None

# def init_logging():
#     logger = logging.getLogger("SAS_RAG_chatbot_openai")
#     if logger.handlers:
#         return
#     logger.propagate = False
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s - %(message)s")
#     handler = logging.FileHandler('sas_rag_chatbot.log')
#     handler.setLevel(logging.INFO)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

if __name__ == "__main__":
    # init_logging()
    logger = logging.getLogger("SAS_RAG_chatbot_openai")
    main()