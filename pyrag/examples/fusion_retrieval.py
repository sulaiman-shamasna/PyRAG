import os
import sys
import argparse
from dotenv import load_dotenv
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from pyrag.utils.helper_functions import replace_t_with_space, show_context
from pyrag.evaluation.evaluate_rag import *

def encode_pdfs_and_get_split_documents(pdf_paths, chunk_size=1000, chunk_overlap=200):
    """
    Encodes PDF files into a vector store using OpenAI embeddings.

    Args:
        pdf_paths: A list of paths to the PDF files.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded PDF content and the cleaned texts.
    """

    all_documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(all_documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    Args:
        documents (List[Document]): List of documents to index.

    Returns:
        BM25Okapi: An index that can be used for BM25 scoring.
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
        vectorstore (VectorStore): The vectorstore containing the documents.
        bm25 (BM25Okapi): Pre-computed BM25 index.
        query (str): The query string.
        k (int): The number of documents to retrieve.
        alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
        List[Document]: The top k documents based on the combined scores.
    """
    # Step 1: Get all documents from the vectorstore
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

    # Step 5: Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores  

    # Step 6: Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 7: Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]

def main():
    parser = argparse.ArgumentParser(description="Process PDF files and perform fusion retrieval.")
    parser.add_argument('--folder', type=str, default='data/', help='Folder containing PDF files')
    parser.add_argument('--query', type=str, default='What is the attention mechanism?', help='Query string for retrieval')
    parser.add_argument('--k', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for vector search scores')
    args = parser.parse_args()

    pdf_folder = args.folder
    query = args.query
    k = args.k
    alpha = args.alpha

    pdf_paths = [os.path.join(pdf_folder, filename) for filename in os.listdir(pdf_folder) if filename.lower().endswith('.pdf')]

    if not pdf_paths:
        print(f"No PDF files found in folder {pdf_folder}")
        sys.exit(1)

    vectorstore, cleaned_texts = encode_pdfs_and_get_split_documents(pdf_paths)
    bm25 = create_bm25_index(cleaned_texts)

    top_docs = fusion_retrieval(vectorstore, bm25, query, k=k, alpha=alpha)
    docs_content = [doc.page_content for doc in top_docs]
    show_context(docs_content)

if __name__ == "__main__":
    main()
