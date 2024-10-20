from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.

    Args:
    original_query (str): The original user query

    Returns:
    str: The rewritten query
    """
    re_write_llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)

    # Create a prompt template for query rewriting
    query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query about transformers and attention mechanisms in machine learning, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

    query_rewrite_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=query_rewrite_template
    )

    # Create an LLMChain for query rewriting
    query_rewriter = query_rewrite_prompt | re_write_llm

    response = query_rewriter.invoke(original_query)
    return response.content.strip()

def generate_step_back_query(original_query):
    """
    Generate a step-back query to retrieve broader context.

    Args:
    original_query (str): The original user query

    Returns:
    str: The step-back query
    """
    step_back_llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)

    # Create a prompt template for step-back prompting
    step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query about transformers and attention mechanisms in machine learning, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

    step_back_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=step_back_template
    )

    # Create an LLMChain for step-back prompting
    step_back_chain = step_back_prompt | step_back_llm

    response = step_back_chain.invoke(original_query)
    return response.content.strip()

def decompose_query(original_query):
    """
    Decompose the original query into simpler sub-queries.

    Args:
    original_query (str): The original complex query

    Returns:
    List[str]: A list of simpler sub-queries
    """
    sub_query_llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)

    # Create a prompt template for sub-query decomposition
    subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query about transformers and attention mechanisms in machine learning, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Example:
Original query: How do transformers work in natural language processing?

Sub-queries:
1. What is the architecture of transformer models in NLP?
2. How does the self-attention mechanism function in transformers?
3. What are the advantages of transformers over RNNs in NLP tasks?
4. How are transformers applied in language translation models?

Now, decompose the following query:

Original query: {original_query}

Sub-queries:"""

    subquery_decomposition_prompt = PromptTemplate(
        input_variables=["original_query"],
        template=subquery_decomposition_template
    )

    # Create an LLMChain for sub-query decomposition
    subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

    response = subquery_decomposer_chain.invoke(original_query).content.strip()
    # Extract the sub-queries from the response
    sub_queries = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('Sub-queries:')]
    return sub_queries

def main():
    # Example query about transformers and attention mechanisms
    original_query = "How do transformers utilize attention mechanisms to process sequential data?"

    # Query Rewriting
    rewritten_query = rewrite_query(original_query)
    print("Original query:", original_query)
    print("\nRewritten query:", rewritten_query)

    # Step-back Prompting
    step_back_query = generate_step_back_query(original_query)
    print("\nStep-back query:", step_back_query)

    # Sub-query Decomposition
    sub_queries = decompose_query(original_query)
    print("\nSub-queries:")
    for i, sub_query in enumerate(sub_queries, 1):
        print(f"{i}. {sub_query}")

if __name__ == '__main__':
    main()
