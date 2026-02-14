import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # Update: Added loader_kwargs to handle encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"} # This is the magic line
    )
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")   
    
    return documents    

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents using the more robust Recursive splitter"""
    print("Splitting documents into chunks...")
    
    # RecursiveCharacterTextSplitter is much more reliable than CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        add_start_index=True # Helpful for tracking where chunks came from
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
    
    # Ensure OPENAI_API_KEY is in your .env file
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store saved to {persist_directory}")
    return vectorstore

def main():
    try:
        documents = load_documents(docs_path='docs')
        chunks = split_documents(documents=documents)
        vectorstore = create_vector_store(chunks=chunks)
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()