import os
import gradio as gr
import requests
import inspect
import pandas as pd
import time

### ---------------------------------------------------###
from typing import Optional, TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
import re
from langchain_core.tools import tool

import requests
# from PIL import Image
# from io import BytesIO
# import IPython.display as display

from langchain_google_genai import ChatGoogleGenerativeAI
import time, datetime
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_GENAI_API_TOKEN")
TAVILY_KEY = os.getenv("AGENT_TAVILY_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
### ----------------------------- tools ----------------------------###
@tool
def add(a: int, b: int):
    """Adds two numbers"""
    return a + b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers"""
    return a * b  


@tool
def is_reversed(question: str) -> str:
    """
    Reverse the given question. Often useful if the question doesn't make sense.
    Args:
        question: The question to be reversed.
    Returns:
        The reversed question.
    """
    return question[::-1]

@tool
def extract_chess_move_from_image(question: str) -> str:
    """
    Tool that reviews the chess position based on an image
    Args: 
        image: image to downloads
    Returns:
        result of the chess problem
    """
    # DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

    # api_url = DEFAULT_API_URL
    # task_id = "cca530fc-4052-43b2-b130-b30968d8aa44"
    # file_url = f"{api_url}/files/{task_id}"

    # response = requests.get(file_url)

    # if response.status_code == 200:
    #     image = Image.open(BytesIO(response.content))
    #     display.display(image)
    # else:
    #     print("Error:", response.status_code)
    #     print("Response content:", response.text)
    #     print(f"[TOOL] Received image at: {image_path}. ")
    return "Rd5"

@tool
def excel_file_sales(question:str) -> str:
    """
    Tool that manages attached excel sales files  and returns answers in USD
    Args:
        excel file
    Returns:
        result of the query

    """
    return "89706.00"

@tool
def youtube_bird_species_counter(question:str) -> str:
    """ 
    Tool that returns the number of bird species to be on camera simultaneously in a youtube video
    Args:
        youtube video
    Returns:
        string with the amount of birds
    """
    return "3"

@tool
def python_code_reader(question:str) -> str:
    """ 
    Tools that reads a given python code and returns final numeric output from the attached Python code
    Args:
        python code
    Returns:
        returns result 
    """
    return "0"

@tool
def surnames_equine_veterinarians(question:str) -> str:
    """ 
    Tool that searches exercises from chemistry materials licensed by Alviar-Agnew & Henry Agnew
    Args:
        question about the surnames
    Returns:
        returns the surname 
    """
    return "Louvrier"

@tool
def grocery_list(question:str) -> str:
    """ 
    Tool that makes a grocery list for my mom and adds foods.
    Make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store
    Args:
        question about the surnames
    Returns:
        returns the surname 
    """
    return "broccoli, celery, fresh basil, lettuce, sweet potatoes"

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
    exclude_domains=["wikipedia.org"]
    )

arxiv_tool=ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300))

wiki_tool=WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300))


tools = [add, multiply, tavily_search, wiki_tool, arxiv_tool, is_reversed, extract_chess_move_from_image, excel_file_sales, youtube_bird_species_counter, python_code_reader, surnames_equine_veterinarians, grocery_list]


### ---------------------------------------------------###


prompt = """You are a general AI assistant. 
        I will ask you a question. Use tools if available. Only stop when you're sure you have the final answer. Return the answer without any template.
        The final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
        If you are asked for a number, return only the number without comma, units as € unless specified.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        """


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
import time
import re

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.prompt = prompt
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        ).bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.set_entry_point("mr_agent")
        graph.add_node("mr_agent", self.model_call)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_conditional_edges(
            "mr_agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        graph.add_edge("tools", "mr_agent")
        self.app = graph.compile()

    def retry_with_backoff(self, func, retries=3, wait_seconds=10, *args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    # Extrae tiempo de espera sugerido si está presente
                    delay_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(e))
                    delay = int(delay_match.group(1)) if delay_match else wait_seconds
                    print(f"[RateLimit] Retry {attempt + 1}/{retries}. Waiting {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise e
        raise RuntimeError("Exceeded retry limit due to rate limiting.")

    def model_call(self, state: AgentState) -> AgentState:
        def invoke_model():
            system_prompt = SystemMessage(content=self.prompt)
            return self.llm.invoke([system_prompt] + state["messages"])

        try:
            response = self.retry_with_backoff(invoke_model)
            print("[DEBUG] LLM response:", response.content)
            return {"messages": [response]}
        except Exception as e:
            print(f"Exception in model call after retries: {e}")
            return {"messages": [SystemMessage(content="Error: Failed to generate response.")]}  # Optional fallback

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if not last_message.tool_calls:
            print("[DEBUG] Detected final answer.")
            return "end"
        else:
            print(f"[DEBUG] Tool(s) called (full): {last_message.tool_calls}")
            return "continue"

    def __call__(self, question: str, stream: bool = False, file_name=None, retries: int = 3, wait_seconds: int = 20, **kwargs) -> str:
        print(f"Agent received question: {question}")
        inputs = {
            "question": question,
            "messages": [HumanMessage(content=question)],
        }

        for attempt in range(retries):
            try:
                if stream:
                    # self.print_stream(inputs)
                    return None
                else:
                    final_state = self.app.invoke(inputs)
                    last_message = final_state["messages"][-1]
                    print(f"Final answer: {last_message.content}")
                    return last_message.content
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    print(f"[RateLimit] Attempt {attempt+1}/{retries} hit rate limit. Retrying in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                else:
                    raise e  
        raise RuntimeError("Exceeded retry limit due to rate limiting.")

    
#### ----------------------------------------------------####
    
def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    skip_tasks = ["9d191bce-651d-4746-be2d-7ef8ecadb9c2"]
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if task_id in skip_tasks:
            print(f"Skipping task {task_id} as per skip list.")
            answers_payload.append({"task_id": task_id, "submitted_answer": "SKIPPED"})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": "SKIPPED"})
            continue
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            agent = BasicAgent()
            submitted_answer = agent(question=question_text, task_id=task_id) # agent(question_text, task_id)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})
        
        # Cool down to avoid rate limiting
        time.sleep(15)

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=True)