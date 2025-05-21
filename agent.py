
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import tempfile
import re
import json
import requests
from urllib.parse import urlparse
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cmath
import pandas as pd
import uuid
import numpy as np
from datetime import datetime
import pytz
import pytesseract


"""Langraph"""
from langgraph.graph import START, StateGraph, MessagesState

from langgraph.prebuilt import ToolNode, tools_condition

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool

from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader


load_dotenv()


### =============== INFORMATION RETRIEVAL TOOLS =============== ###

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return a maximum of 3 results.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(
        max_results=3, 
        include_raw_content=True,
        include_images=True,
        exclude_domains = ["wikipedia.org"]
        ).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return a maximum of 3 results.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}



@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return a maximum of 3 results.
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=3, lang = "en").load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}




# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
print(system_prompt)

# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)  #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents2",
    query_name="match_documents_2",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)


tools = [
    web_search,
    arxiv_search,
    wiki_search,
]


# Build graph function
def build_graph(provider: str):
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            task="text-generation", 
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
            temperature=0,
        ),
        verbose=True,
    )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)

        if similar_question:  # Check if the list is not empty
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
            )
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        else:
            # Handle the case when no similar questions are found
            return {"messages": [sys_msg] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()



