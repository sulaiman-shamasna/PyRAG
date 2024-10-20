import os
import sys
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print("Error: OPENAI_API_KEY not found. Please set it in the .env file.")
    sys.exit(1)

# Set the API key explicitly in os.environ
os.environ["OPENAI_API_KEY"] = api_key

# Import LangChain and related modules
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field

# Define Pydantic models for structured outputs
class CategoryOptions(BaseModel):
    category: str = Field(
        description="The category of the query, the options are: Factual, Analytical, Opinion, or Contextual",
        example="Factual"
    )

class RelevantScore(BaseModel):
    score: float = Field(
        description="The relevance score of the document to the query",
        example=8.0
    )

class SelectedIndices(BaseModel):
    indices: List[int] = Field(
        description="Indices of selected documents",
        example=[0, 1, 2, 3]
    )

class SubQueries(BaseModel):
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
        example=["What is the population of New York?", "What is the GDP of New York?"]
    )

# Query Classifier to determine the category of the query
class QueryClassifier:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0, max_tokens: int = 4000):
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\n"
                "Query: {query}\nCategory:"
            )
        )
        # Define the output parser
        parser = PydanticOutputParser(pydantic_object=CategoryOptions)
        # Create the LLMChain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=parser)

    def classify(self, query: str) -> str:
        print("Classifying query...")
        result = self.chain.run(query)
        print(f"Classified Category: {result}")
        return result

# Base Retrieval Strategy
class BaseRetrievalStrategy:
    def __init__(self, texts: List[str], model_name: str = "gpt-4", temperature: float = 0.0, max_tokens: int = 4000):
        self.embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
        self.documents = text_splitter.create_documents(texts)
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)

# Factual Retrieval Strategy
class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        print("Retrieving factual information...")
        # Enhance the query for better information retrieval
        enhance_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        parser = PydanticOutputParser(pydantic_object=CategoryOptions)  # Adjust as needed
        enhance_chain = LLMChain(llm=self.llm, prompt=enhance_prompt)
        enhanced_query = enhance_chain.run(query)
        print(f"Enhanced Query: {enhanced_query}")

        # Retrieve documents using the enhanced query
        docs = self.db.similarity_search(enhanced_query, k=k*2)

        # Rank the relevance of retrieved documents
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template=(
                "On a scale of 1-10, how relevant is this document to the query: '{query}'?\n"
                "Document: {doc}\nRelevance score:"
            )
        )
        ranking_parser = PydanticOutputParser(pydantic_object=RelevantScore)
        ranking_chain = LLMChain(llm=self.llm, prompt=ranking_prompt, output_parser=ranking_parser)

        ranked_docs = []
        print("Ranking documents...")
        for doc in docs:
            score_str = ranking_chain.run({"query": enhanced_query, "doc": doc.page_content})
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]

# Analytical Retrieval Strategy
class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        print("Retrieving analytical information...")
        # Generate sub-queries for comprehensive analysis
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )
        sub_queries_parser = PydanticOutputParser(pydantic_object=SubQueries)
        sub_queries_chain = LLMChain(llm=self.llm, prompt=sub_queries_prompt, output_parser=sub_queries_parser)
        sub_queries = sub_queries_chain.run({"query": query, "k": k}).sub_queries
        print(f"Sub-queries: {sub_queries}")

        all_docs = []
        for sub_query in sub_queries:
            all_docs.extend(self.db.similarity_search(sub_query, k=2))

        # Select the most diverse and relevant set of documents
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Select the most diverse and relevant set of {k} documents for the query: '{query}'\n"
                "Documents:\n{docs}\nSelected indices:"
            )
        )
        diversity_parser = PydanticOutputParser(pydantic_object=SelectedIndices)
        diversity_chain = LLMChain(llm=self.llm, prompt=diversity_prompt, output_parser=diversity_parser)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        selected_indices_result = diversity_chain.run({"query": query, "docs": docs_text, "k": k})
        indices = selected_indices_result.indices
        print(f"Selected document indices: {indices}")
        return [all_docs[i] for i in indices if i < len(all_docs)]

# Opinion Retrieval Strategy
class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        print("Retrieving opinion-based information...")
        # Identify distinct viewpoints
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = LLMChain(llm=self.llm, prompt=viewpoints_prompt)
        viewpoints = viewpoints_chain.run({"query": query, "k": k}).split('\n')
        viewpoints = [v.strip() for v in viewpoints if v.strip()]
        print(f"Viewpoints: {viewpoints}")

        all_docs = []
        for viewpoint in viewpoints:
            all_docs.extend(self.db.similarity_search(f"{query} {viewpoint}", k=2))

        # Classify and select diverse opinions
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template=(
                "Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\n"
                "Documents:\n{docs}\nSelected indices:"
            )
        )
        opinion_parser = PydanticOutputParser(pydantic_object=SelectedIndices)
        opinion_chain = LLMChain(llm=self.llm, prompt=opinion_prompt, output_parser=opinion_parser)
        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        selected_indices_result = opinion_chain.run({"query": query, "docs": docs_text, "k": k})
        indices = selected_indices_result.indices
        print(f"Selected document indices: {indices}")
        return [all_docs[i] for i in indices if i < len(all_docs)]

# Contextual Retrieval Strategy
class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def retrieve(self, query: str, k: int = 4, user_context: str = None) -> List[Document]:
        print("Retrieving contextual information...")
        # Incorporate user context into the query
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the user context: {context}\n"
                "Reformulate the query to best address the user's needs: {query}"
            )
        )
        contextualized_query = self.llm(
            prompt=context_prompt.render(query=query, context=user_context or "No specific context provided")
        ).content
        print(f"Contextualized Query: {contextualized_query}")

        # Retrieve documents using the contextualized query
        docs = self.db.similarity_search(contextualized_query, k=k*2)

        # Rank the relevance of retrieved documents considering the user context
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template=(
                "Given the query: '{query}' and user context: '{context}', "
                "rate the relevance of this document on a scale of 1-10:\n"
                "Document: {doc}\nRelevance score:"
            )
        )
        ranking_parser = PydanticOutputParser(pydantic_object=RelevantScore)
        ranking_chain = LLMChain(llm=self.llm, prompt=ranking_prompt, output_parser=ranking_parser)

        ranked_docs = []
        print("Ranking documents...")
        for doc in docs:
            score_str = ranking_chain.run({
                "query": contextualized_query,
                "context": user_context or "No specific context provided",
                "doc": doc.page_content
            })
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            ranked_docs.append((doc, score))

        # Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]

# Adaptive Retriever to select the appropriate strategy
class AdaptiveRetriever:
    def __init__(self, texts: List[str]):
        self.classifier = QueryClassifier()
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }

    def get_relevant_documents(self, query: str, user_context: str = None) -> List[Document]:
        category = self.classifier.classify(query)
        print(f"Query classified as: {category}")
        strategy = self.strategies.get(category)
        if not strategy:
            print(f"No strategy found for category '{category}'. Using FactualRetrievalStrategy by default.")
            strategy = self.strategies["Factual"]
        if category == "Contextual":
            return strategy.retrieve(query, user_context=user_context)
        return strategy.retrieve(query)

# Pydantic Adaptive Retriever (if needed for integration)
class PydanticAdaptiveRetriever:
    def __init__(self, adaptive_retriever: AdaptiveRetriever):
        self.adaptive_retriever = adaptive_retriever

    def get_relevant_documents(self, query: str, user_context: str = None) -> List[Document]:
        return self.adaptive_retriever.get_relevant_documents(query, user_context)

# Adaptive RAG System
class AdaptiveRAG:
    def __init__(self, texts: List[str]):
        adaptive_retriever = AdaptiveRetriever(texts)
        self.retriever = PydanticAdaptiveRetriever(adaptive_retriever=adaptive_retriever)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)
        
        # Initialize memory
        self.memory = ConversationBufferMemory()

        # Create a custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # Define the output parser (assuming plain text response)
        # If you need structured output, define a Pydantic model and use PydanticOutputParser
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def answer(self, query: str) -> str:
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Incorporate memory
        self.memory.save_context({"question": query}, {"context": context})
        combined_context = self.memory.load_memory_variables({})["history"] + "\n" + context

        # Generate the answer
        input_data = {"context": combined_context, "question": query}
        answer = self.llm_chain.run(input_data)
        return answer

# Function to print answers nicely
def print_answer(category: str, answer: str):
    print(f"\n===== {category} =====")
    print(f"{answer}\n{'='*50}\n")

# Main function to handle user interaction
def main():
    # Sample texts for the vector store
    texts = [
        "The Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
        "New York City is the most populous city in the United States.",
        "The GDP of New York is one of the highest among US states.",
        "Climate change is significantly affecting global weather patterns.",
        "There are multiple theories about the origin of life on Earth, including primordial soup and hydrothermal vents.",
        "Earth's position in the Solar System plays a crucial role in its habitability."
    ]

    # Initialize the Adaptive RAG system
    rag_system = AdaptiveRAG(texts)

    print("Welcome to the Adaptive RAG System! You can ask questions related to Earth and its environment.")
    print("Type 'exit' to quit the conversation.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid question.")
            continue

        # Get the answer from the RAG system
        answer = rag_system.answer(user_query)

        # Classify the query to determine category for better formatting
        classifier = QueryClassifier()
        category = classifier.classify(user_query)

        # Print the answer nicely
        print_answer(category, answer)

if __name__ == "__main__":
    main()
