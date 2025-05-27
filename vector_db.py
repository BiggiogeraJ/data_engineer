from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from doc_loader import load_documents
from doc_chunker import split_documents
from embedder import get_embedder
import os
import argparse
import shutil
import sys

db_location = "./chroma_db_de"


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks



def create_chroma(chunks: list[Document], path: str, embed_function: callable = None):

    # Load the existing database.
    db = Chroma(
        persist_directory=path, embedding_function=embed_function
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    if len(chunks):
        print(f"👉 Adding new documents: {len(chunks)}")
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        db.add_documents(chunks_with_ids, ids=chunk_ids)
    else:
        print("✅ No new documents to add")

def update_chroma(chunks: list[Document], path: str, embed_function: callable = None):
    # Load the existing database.
    db = Chroma(
        persist_directory=path, embedding_function=embed_function
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    print(len(new_chunks), "new chunks to add")

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def clear_database(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create the database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--update", action="store_true", help="Update the database if you have added new documents or new versions of existing documents.")

    args = parser.parse_args()
    if not (args.create or args.reset or args.update):
        print("Please specify an action, use --help for the list of available actions.")
        sys.exit()

    # Load documents from the specified directory
    documents = load_documents(PATH_TO_PDF_DIR="./data")
    
    # Split the loaded documents into smaller chunks
    split_docs = split_documents(documents)
    
    # Initialize the embedder for embedding text
    embedder = get_embedder()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database(db_location)
        create_chroma(split_docs, db_location, embedder)
    
    elif args.create:
        if os.path.exists(db_location):
            print(f"⚠️  Database already exists at {db_location}. If you want to reset it, please call the --reset argument.")
            sys.exit()
        # Create the database if it does not exist.
        print("✨ Creating Database")
        create_chroma(split_docs, db_location, embedder)

    elif args.update:
        print("✨ Updating Database")
        update_chroma(split_docs, db_location, embedder)


if __name__ == "__main__":
    main()