from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from embedder import get_embedder
from vector_db import retriever
import argparse



CHROMA_PATH = "chroma_db_de"

PROMPT_TEMPLATE = """
You are an helpful assistant who is an expert in data engineering and mining information intelligence.
Answer the question using your broad wealth of knoweldge but primarily based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedder()

    # Search the DB.
    results = retriever.invoke(query_text)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.2", temperature=0.1)
    response_text = model.invoke(prompt)

    sources = [" ".join([doc.metadata.get("source", None), ', page: '+str(doc.metadata.get("page", None))]) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

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