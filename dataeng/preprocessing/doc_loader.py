from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(PATH_TO_PDF_DIR="./data"):
    # Load documents from the specified directory
    loader = PyPDFDirectoryLoader(PATH_TO_PDF_DIR)
    documents = loader.load()
    
    # Return the loaded documents
    return documents

if __name__ == "__main__": 
    # Load documents and print the number of loaded documents
    documents = load_documents(PATH_TO_PDF_DIR="./data/data_engineering_mining_information_intelligence.pdf")
    print(f"Loaded {len(documents)} documents.")
    
    # Print the first document's metadata
    if documents:
        print("First document metadata:", documents[-1].metadata)
        print("First document page content:", documents[-1].page_content)
    else:
        print("No documents loaded.")