from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
import uuid
import logging
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from tools import wikipedia_search_html, website_scrape, web_search, visual_model, audio_model, run_python, data_tool
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

def get_graph():
    """Build and return a LangGraph for a conversational agent with tools."""
        
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    
    # Model configuration
    llm_model = "gemini-2.0-flash"  # Options: "gemma-3-27b-it", "gemini-2.0-flash-lite"

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    llm_gemma = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )

    # Define tools
    tools = [web_search, wikipedia_search_html, website_scrape, visual_model, audio_model, run_python, data_tool]
    
    # Bind tools to the LLM
    chat_with_tools = llm.bind_tools(tools)

    # Define the state type with annotations
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        last_ai_message: Optional[str]
        question: Optional[str]
        final_answer: Optional[str]
        error: Optional[str]  # Added error field for tracking issues

    # Assistant node - generates responses
    def assistant(state: AgentState) -> Dict[str, Any]:
        """Generate a response using the LLM."""
        try:
            logger.info("Assistant node processing")
            # Check if the last message is from the assistant
            # prompt = f"Given the following question: {state['question']} Answer it using the tools available."
            # if state["planner_message"]:
            #     prompt += f"\nHere is a Plan to help you answer the user question: {state['planner_message']}"
            response = chat_with_tools.invoke(state["messages"])
            return {
                "messages": [response],
                "last_ai_message": response.content #if state["messages"] and isinstance(state["messages"][-1], AIMessage) else None
            }
        except Exception as e:
            logger.error(f"Error in assistant node: {e}")
            return {
                "error": f"Assistant error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while generating a response. Please try again.")]
            }


    # Error handling node
    def handle_error(state: AgentState) -> Dict[str, Any]:
        """Handle errors in the graph execution."""
        error_msg = state.get("error", "Unknown error")
        logger.error(f"Handling error: {error_msg}")
        return {
            "messages": [AIMessage(content=f"I apologize, but I encountered an error: {error_msg}. Please try again or rephrase your question.")]
        }
    
    
    # Define your desired data structure.
    class AnswerTemplate(BaseModel):
        thought: str = Field(description="Thought process of the model")
        answer: str = Field(description="Final answer to the question")

    def validate_answer(state: AgentState) -> Dict[str, Any]:
        """Validate the final answer."""
        try:
            logger.info("Validating answer")
            
            llm_gemma = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                temperature=0,
                max_tokens=None,
                timeout=60,  # Added a timeout
                max_retries=2,
            )

            def escape_braces(text):
                return text.replace("{", "{{").replace("}", "}}")

            query = "You are given an interaction between human and AI agent. Format the AGENT ANSWER in json with the following keys: answer. Answer should be the final answer from the AGENT."
            
            # Set up a parser + inject instructions into the prompt template.
            parser = JsonOutputParser(pydantic_object=AnswerTemplate)
            prompt = PromptTemplate(
                        template=(
                            f"SYSTEM MESSAGE: {SYSTEM_MESSAGE}\n\n"
                            f"HUMAN QUERY: {escape_braces(state['question'])}\n\n"
                            f"AGENT ANSWER: {escape_braces(state['last_ai_message'])}\n\n" 
                            # "{format_instructions}\n{query}"
                            f"{query}"
                        ),
                        input_variables=["query"],
                        # partial_variables={"format_instructions": parser.get_format_instructions()},
                    )
            chain = prompt | llm_gemma | parser

            return {
                "final_answer": chain.invoke({"query": query})["answer"]
            }
        except Exception as e:
            logger.error(f"Error in validate_answer node: {e}")
            return {
                "error": f"Validation error: {str(e)}",
                "messages": [AIMessage(content="I encountered an error while validating the answer. Please try again.")]
            }

    # Build the graph
    builder = StateGraph(AgentState)

    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("validate_answer", validate_answer)
    # builder.add_node("handle_error", handle_error)

    # Define edges for the standard flow
    builder.add_edge(START, "assistant")

    # Conditional edges from assistant
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",  # Route to tools if needed
            END: "validate_answer"  # Route to validate_answer if no tools needed
        }
    )
    
    # From tools back to assistant to process tool results
    builder.add_edge("tools", "assistant")
    builder.add_edge("validate_answer", END)

    # Set up memory for conversation persistence
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    

    
    return graph

class BasicAgent:
    """A simple agent that manages interaction with the LangGraph."""
    
    def __init__(self, graph=None):
        """Initialize the agent with a LangGraph."""
        if graph is None:
            self.graph = get_graph()
        else:
            self.graph = graph
        logger.info("BasicAgent with LangGraph initialized.")

        thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Agent thread ID: {thread_id}")

    def __call__(self, question: str, task_id: str) -> str:
        """Process a question through the agent and return the response."""
        logger.info(f"Agent received question: {question[:50]}...")
        # Create a system message to guide the model's behavior
        system_message = SystemMessage(
            content=SYSTEM_MESSAGE
        )
        try:
            # Construct initial state
            question += f"\n task_id={task_id}"
            initial_state = {
                "messages": [system_message, HumanMessage(content=question)],
                # "last_ai_message": None,
                "question": question,
                "error": None
            }

            # Run the LangGraph
            final_state = self.graph.invoke(initial_state, self.config)
            # Check for errors in the final state
            if "error" in final_state and final_state["error"] is not None:
                logger.error(f"Error in final state: {final_state['error']}")
            
            final_answer = final_state.get("final_answer", None)

            if final_answer:
                # If a final answer is available, return it
                logger.info(f"Agent returning final answer: {final_answer}")
                return final_answer

            # Fallback if no AI message found
            fallback = "Sorry, I could not generate a response."
            logger.warning("Agent fallback response - no AI message found")
            return fallback
            
        except Exception as e:
            logger.error(f"Unhandled exception in agent: {e}", exc_info=True)
            return f"Sorry, I encountered an unexpected error: {str(e)}"

# Example usage
if __name__ == "__main__":
    agent = BasicAgent()

    response = agent(question="""The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.task_id=7bd855d8-463d-4ed5-93ca-5fe35145f733""", task_id="7bd855d8-463d-4ed5-93ca-5fe35145f733")
    
    print(f"Response: {response}")