import sys
sys.path.append("/Users/jacopo.biggiogera@igenius.ai/Desktop/GenAI_projects/data_engineer")
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from dataeng.preprocessing.doc_loader import load_documents
from dataeng.preprocessing.doc_chunker import split_documents
from dataeng.preprocessing.embedder import get_embedder
from dataeng.vectordb.dbutils import create_chroma, update_chroma, clear_database
import os
import argparse

db_location = "./chroma_db_de"


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