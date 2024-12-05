import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_chroma import Chroma  
import INIT_MODEL as md
import json

# 將檢索到的文檔內容格式化為單一文本
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 建立完整的 RAG（檢索-生成）鏈
def chain(load_path=None):
    # 初始化模型和嵌入函數
    llm = md.llm  # 語言模型
    embeddings = md.embeddings  # 向量嵌入模型

    retrievers = []  # 儲存多個檢索器
    for path in load_path:
        db_load = Chroma(persist_directory=path, embedding_function=embeddings)  # 加載向量數據庫
        retrievers.append(db_load.as_retriever(search_kwargs={"k": 50}))  # 將數據庫轉換為檢索器

    # 定義對話提示模板
    template = """
    你是一個專門回答化學領域問題的專家，你的任務是根據上下文的內容來回答使用者提出的問題。
    所有回答都必須依據提供的資料來源。如果無法在資料來源中找到答案，請明確表示你不知道答案。注意以下幾點：

    1. 你只能回答與化學物質或化學相關的問題，對於非化學相關的問題，請回答「此問題無法回答，請詢問化學相關問題」。
    2. 僅回答當前的問題，並且不要重複之前已經回答過的問題。
    3. 如果不知道答案，請明確回答「依據目前的資料，無法回答此問題」，不要生成任何不確定的或無關的答案。
    4. 你的回答必須完全基於資料來源，不應推測或引入額外的資訊。
    5. 所有回答都必須使用繁體中文。
    6. 使用敘述的方式回答問題。

    資料來源：{context}
    問題：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)  # 初始化提示模板
    config = RailsConfig.from_path("./config")  # 加載 Guardrails 配置

    # 多數據庫檢索器：從多個向量數據庫中檢索相關文件
    def multi_db_retrieve(query, retriever_result="retriever_result.json"):
        all_results = []  # 儲存所有檢索結果
        merged_contexts = {}  # 合併上下文內容
        current_index = 1  # 索引遞增，用於唯一鍵值

        for retriever in retrievers:
            results = retriever.invoke(query)  # 執行檢索
            print(f"Retrieved {len(results)} documents from a retriever:")
            for i, doc in enumerate(results):
                print(f"Document {i+1}: {doc.page_content[:200]}...")  # 打印文件前 200 個字元
            
            # 合併文件內容
            for doc in results:
                key = f"context_{current_index}"  # 使用索引作為鍵
                merged_contexts[key] = doc.page_content
                current_index += 1

            all_results.extend(results)  # 添加到總結果

        # 將合併的上下文保存為 JSON 文件
        with open(retriever_result, "w", encoding="utf-8") as file:
            json.dump([merged_contexts], file, ensure_ascii=False, indent=4)

        return all_results

    # 格式化多數據庫檢索的文檔
    def format_combined_docs(query):
        docs = multi_db_retrieve(query)  # 執行檢索
        formatted_docs = format_docs(docs)  # 格式化文檔
        return formatted_docs

    # 定義處理鏈：格式化上下文、處理問題、生成回應
    chain = (
        {"context": format_combined_docs, "question": RunnablePassthrough()}  # 提取上下文和問題
        | prompt  # 構建提示
        | llm  # 語言模型生成回答
        | StrOutputParser()  # 解析生成的回應
    )

    # 處理異步事件循環
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 整合 Guardrails
    guardrails = RunnableRails(config)  # 加載配置
    chain_with_guardrails = guardrails | chain  # 將 Guardrails 與處理鏈結合

    return chain_with_guardrails  # 返回完整處理鏈