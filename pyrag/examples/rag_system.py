"""
This is a naive chatbot using the LangChain framework. It can be used as a base to build more sophisticated chatbots.
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA

# Load environment variables from the main directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def main():
    parser = argparse.ArgumentParser(description='Chatbot with RAG over PDF files')
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing PDF files')
    args = parser.parse_args()

    # Resolve the folder path relative to the current working directory
    folder_path = os.path.abspath(args.folder)
    if not os.path.isdir(folder_path):
        print(f"The specified folder does not exist: {folder_path}")
        sys.exit(1)

    # List and print PDF files found
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in the specified folder: {folder_path}")
        sys.exit(1)
    print(f"PDF files found: {pdf_files}")

    # Load PDF documents
    documents = []
    for filename in pdf_files:
        filepath = os.path.join(folder_path, filename)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            print(f"Loaded {len(docs)} pages from {filename}.")
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} document chunks.")

    # Create embeddings and store in vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Vectorstore contains {vectorstore.index.ntotal} documents.")

    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize LLM
    llm = ChatOpenAI(temperature=0)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Define a tool that uses the QA chain
    def qa_tool(input_text):
        return qa_chain.run(input_text)

    tools = [
        Tool(
            name="PDF Knowledge Base",
            func=qa_tool,
            description="Use this tool to answer questions based on the PDFs."
        )
    ]

    # Initialize agent with memory
    agent_chain = initialize_agent(
        tools,
        llm,
        agent="chat-conversational-react-description",
        verbose=True,
        memory=memory
    )

    # Start chat
    print("Welcome to the PDF Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = agent_chain.run(input=user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
