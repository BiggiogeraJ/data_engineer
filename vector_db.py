from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from doc_loader import load_documents
from doc_chunker import split_documents
from embedder import get_embedder
import os

db_location = "./chroma_db_de"
add_docs = not os.path.exists(db_location)

# Load documents from the specified directory
documents = load_documents(PATH_TO_PDF_DIR="./data")
    
# Split the loaded documents into smaller chunks
split_docs = split_documents(documents)
    
# Initialize the embedder for embedding text
embedder = get_embedder()

vector_db = Chroma.from_documents(collection_name = 'data_engineering', embedding_function = embedder, persist_directory=db_location)

if add_docs:
    docs = []
    ids = []
    for i in range(len(split_docs)):
        # Add id to each document
        doc = Document(page_content=split_docs[i].page_content,
                        metadata=split_docs[i].metadata,
                        id = str(i))
        docs.append(doc)
        ids.append(str(i))
        

    # add documents to the vector database
    vector_db.add_documents(docs, ids=ids)

#Add retreival functionalality to vector store
retriever = vector_db.as_retriever(search_kwargs={"k": 5})