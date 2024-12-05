import requests
import logging
import json
import streamlit as st
import CHAIN as rag
import os

def get_id_from_url():
    chemical_id = st.query_params.id
    if chemical_id:
        return chemical_id  
    else:
        st.warning("No Chemical ID provided in URL")
        return None
    
def load_retrieved_documents_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        contents = [value for key, value in data[0].items()]
        return contents

def save_conversation_to_json(user_input, chatbot_response, retrieved_documents):
    data = {
        "user_input": user_input,
        "retrieved_contexts": retrieved_documents, 
        "response": chatbot_response,
        "reference":"",  
    }
    output_path = "chatbot_response.json"

    try:
        try:
            with open(output_path, 'r') as file:
                conversations = json.load(file)
        except FileNotFoundError:
            conversations = []

        conversations.append(data)

        with open(output_path, 'w') as file:
            json.dump(conversations, file, ensure_ascii=False, indent=4)
        print(f"對話數據已保存到 {output_path}")
    except Exception as e:
        print(f"保存數據時出錯: {e}")


def get_response(query):
    try:
        vector_db_path = "./VECTOR_DB"

        # 判斷查詢的類型
        if is_alternative_query(query):
            load_path = [
                os.path.join(vector_db_path, "SUMMARY_1500"),
                os.path.join(vector_db_path, "CHILDRENS_PRODUCTS"),
            ]
        # elif is_summary_query(query):
        #     # 如果是摘要查詢，只載入 SUMMARY 路徑
        #     load_path = [os.path.join(vector_db_path, "SUMMARY")]
        # else:
        #     # 默認載入多個路徑
        #     load_path = [
        #         os.path.join(vector_db_path, "SUMMARY"),
        #         os.path.join(vector_db_path, "CHILDRENS_PRODUCTS"),
        #         os.path.join(vector_db_path, "CHEMICAL_ALTERNATIVES")
        #     ]
        else:
            load_path = [os.path.join(vector_db_path, "SUMMARY_1500")]

        chain = rag.chain(load_path=load_path)
        response = chain.invoke(query)
        logging.info(f"Response from chain.invoke(): {response}")
        retriever_result = "retriever_result.json"
        retrieved_documents = load_retrieved_documents_from_file(retriever_result)

        if isinstance(response, dict):
            response_text = response.get('output', '')
        else:
            response_text = response

        if response_text.strip() == "I'm sorry, I can't respond to that.":
            response_text = "此問題無法回答，請試著詢問其他化學物質相關問題"

                # 保存對話紀錄
        save_conversation_to_json(
            user_input=query,
            retrieved_documents=retrieved_documents,
            chatbot_response=response_text,
        )

        return response_text

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
    summary_keywords = [
        "總結", "概述", "摘要", "回顧", "重點", "要點", 
        "整理", "概括", "精簡", "簡述", "簡報", 
        "總覽", "報告", "分析報告", "總結陳述", 
        "小結", "概覽", "精要", "精華", "總體描述", 
        "一覽", "提要", "概要", "大意", "歸納", 
        "簡要說明", "總述", "提綱", "說明重點", 
        "重點整理", "精簡總結", "一目了然", "總體分析", 
        "回顧與總結", "大綱", "結論", "關鍵點", 
        "條列整理", "重點摘要", "簡要概述", "整體歸納"
    ]
    return any(keyword in query for keyword in summary_keywords)


def is_alternative_query(query):
    alternative_keywords = [
        "替代物", "化學替代物", "替代品", "替代選項", 
        "替代", "取代", "替換", "替用", "替補", 
        "取代方案", "代用品", "取代的選項", "替代化學品", 
        "可替代品", "可替代物", "代替選擇", "替換方案", 
        "可取代", "取代方案", "替用產品", "替代商品",
        "可替代", "化學品替代", "取而代之", "替代材料",
        "更安全替代品", "環保替代品", "無毒替代品",
        "取代方法", "替代製程", "綠色替代", "安全替代",
        "替代技術", "替代機制"
    ]
    return any(keyword in query for keyword in alternative_keywords)

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
    
if __name__ == "__main__":
    main()