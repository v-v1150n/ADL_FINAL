import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_chroma import Chroma  
import init_model as md
import json

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def chain(load_path=None):
    llm = md.llm
    embeddings = md.embeddings

    retrievers = []
    for path in load_path:
        db_load = Chroma(persist_directory=path, embedding_function=embeddings)
        retrievers.append(db_load.as_retriever(search_kwargs={"k": 30}))

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
    prompt = ChatPromptTemplate.from_template(template)
    config = RailsConfig.from_path("./config")

    def multi_db_retrieve(query, retriever_result = "retriever_result.json"):
        all_results = []
        merged_contexts = {}  # 用來合併所有的 context
        current_index = 1  # 確保索引連續遞增

        for retriever in retrievers:
            results = retriever.invoke(query)
            print(f"Retrieved {len(results)} documents from a retriever:")
            
            # 將每個文件的內容加到合併字典中
            for doc in results:
                key = f"context_{current_index}"  # 使用 current_index 作為鍵
                merged_contexts[key] = doc.page_content
                current_index += 1  # 確保索引連續遞增

            all_results.extend(results)

        # 將合併後的字典保存到 JSON 文件中
        with open(retriever_result, "w", encoding="utf-8") as file:
            json.dump([merged_contexts], file, ensure_ascii=False, indent=4)

        return all_results

    # Format the combined documents from all vector databases
    def format_combined_docs(query):
        docs = multi_db_retrieve(query)  # Retrieve all relevant documents
        formatted_docs = format_docs(docs)  # Format the documents into a single string
        return formatted_docs  # Return the formatted context to be used by the LLM

    chain = (
        {"context": format_combined_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    guardrails = RunnableRails(config) # config.yml
    chain_with_guardrails = guardrails | chain

    return chain_with_guardrails
