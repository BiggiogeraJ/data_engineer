import sys
sys.path.append("/Users/jacopo.biggiogera@igenius.ai/Desktop/GenAI_projects/data_engineer")
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from dataeng.preprocessing.embedder import get_embedder
import argparse



CHROMA_PATH = "chroma_db_de"

PROMPT_TEMPLATE = """
You are an helpful assistant who is an expert in data engineering and mining information intelligence.
Answer the question using your broad wealth of knoweldge and in addition cross check and supplement it with the context provided below:

{context}

If you deem it necessary also add a brief useful example to illustrate your answer.
If the context does not provide enough information to answer the question, say "I don't know" and do not make up an answer.
If the context provides information that is not relevant to the question, do not include it in your answer.
Do not refer to the context explicitly in your answer by saying "according to the context" or similar phrases but rather weave it in to your answer.
---

Answer the question following the context and the instructions provided above: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedder()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    # Search the DB.
    results = retriever.invoke(query_text)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.2", temperature=0.1)
    response_text = model.invoke(prompt)

    sources = [" ".join([doc.metadata.get("source", None), ', page: '+str(doc.metadata.get("page", None))]) for doc in results]
    formatted_response = f"Response: \n{response_text}\nSources: {sources}"
    print('-----------------------------------------------------------------------------')
    print(' ')
    print(formatted_response)
    print(' ')
    print('-----------------------------------------------------------------------------')
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

if __name__ == "__main__":
    main()