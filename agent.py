import os

from typing import TypedDict, Sequence, Dict, Any, Optional, Annotated
from langgraph.graph import stategraph, END, START
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage

from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import logging 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


###------------Set up API keys------------###



from dotenv import load_dotenv
import os
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_GENAI_API_TOKEN")
TAVILY_KEY = os.getenv("AGENT_TAVILY_API_KEY")


###------------Set the Agent State------------###
class AgentState(TypedDict):
    question: Optional[str]
    last_ai_message: Optional[str]
    final_answer: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]

###------------- Set tools --------------------###

tavily_search = TavilySearch(
    tavily_api_key=TAVILY_KEY,
    max_results=5,
    topic="general",
    include_answer=True,
    # include_images=False,
    # include_image_descriptions=False,
    search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

@tool
def add(a:int, b:int):
    """This function adds two given numbers"""
    return a + b

@tool
def multiply(a:int, b:int):
    """This function multiplies two given numbers"""
    return a + b

tools = [add, multiply, tavily_search]

###------------Call the LLM and bind tools------------###

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0,
    google_api_key=GOOGLE_API_KEY
).bind_tools(tools)


###------------Define Nodes of the Agent------------###

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content= 
    """You are a general AI assistant. 
        I will ask you a question. 
        Report your thoughts, and give your FINAL ANSWER directly without using any template. 
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
        If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.""")
    response = llm.invoke([system_prompt] + state["messages"])
    return{"messages":[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)

graph.add_node("mr_agent", model_call)

# tool node is used to attach the tools to the graph
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("mr_agent")

graph.add_conditional_edges(
    "mr_agent", 
    should_continue,

    {
        "continue":"tools",
        "end":END,
    }
)

graph.add_edge("tools", "mr_agent")

app = graph.compile()

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))

