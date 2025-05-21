from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits documents into smaller chunks for easier processing.

    Args:
        documents (list[Document]): List of documents to be split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[Document]: List of split documents.
    """
    # Initialize the text splitter with the specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Split each document into smaller chunks
    split_documents = text_splitter.split_documents(documents)

    return split_documents


