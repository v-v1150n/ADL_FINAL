from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
import os
import json
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


os.environ["OPENAI_API_KEY"] = "Your API KEY"

with open("b_part4.json", "r", encoding="utf-8") as file:
    data = json.load(file)

eval_dataset = EvaluationDataset.from_dict(data)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", timeout=600))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]
results = evaluate(dataset=eval_dataset, metrics=metrics)

df = results.to_pandas()
df.to_csv("./result4.csv", index=False, encoding="utf-8-sig")  