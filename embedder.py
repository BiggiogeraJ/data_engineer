from langchain_ollama import OllamaEmbeddings

def get_embedder():
    """
    Returns an instance of OllamaEmbeddings for embedding text.
    
    Returns:
        OllamaEmbeddings: An instance of the OllamaEmbeddings class.
    """
    # Initialize the OllamaEmbeddings with the specified model
    embedder = OllamaEmbeddings(
        model="nomic-embed-text:v1.5")
    
    return embedder