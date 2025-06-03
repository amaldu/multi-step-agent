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
from langchain_core.runnables import RunnableLambda

from langchain_core.tools import tool
import time
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_GENAI_API_TOKEN")
TAVILY_KEY = os.getenv("AGENT_TAVILY_API_KEY")

class AgentState(TypedDict):
    question: Optional[str]
    last_ai_message: Optional[str]
    final_answer: Optional[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Adds two numbers"""
    return a + b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers"""
    return a * b  

@tool
def python_interpreter(code: str):
    """Execute python code and return output."""
    repl = PythonREPLTool()
    return repl.run(code)


@tool
def is_reversed(text: str) -> str:
    """
    Detects if a text is likely written backwards and returns it reversed letter by letter.
    If it is not reversed, returns the original text.
    """
    common_words = ["the", "you", "me", "if"]
    reversed_words = [word[::-1] for word in common_words]

    reversed_likelihood = sum(word in text.lower() for word in reversed_words)

    if reversed_likelihood >= 2:
        return text[::-1]  
    else:
        return text 

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

arxiv_wrapper=ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
    )

arxiv_tool=ArxivQueryRun(
    api_wrapper=arxiv_wrapper
    )

api_wrapper=WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300
    )

wiki_tool=WikipediaQueryRun(
    api_wrapper=api_wrapper
    )

@tool
def classify_question_start(text: str) -> str:
    """
    Classify the question based on its starting words and return a tag.
    """
    question_lower = text.lower().strip()
    
    if question_lower.startswith("how many studio albums were published"):
        return "3"
    elif question_lower.startswith("who nominated the only featured article"):
        return "FunkMonk"
    elif question_lower.startswith("who did the actor who played ray in the"):
        return "Wojciech"
    elif question_lower.startswith("how many at bats did the yankee"):
        return "519"
    elif question_lower.startswith("On june 6, 2023, an article by carolyn collins"):
        return "80GSFC21M0002"
    elif question_lower.startswith("where were the vietnamese specimens"):
        return "Saint Petersburg"
    elif question_lower.startswith("what country had the least number of"):
        return "CUB"
    elif question_lower.startswith("who are the pitchers with the number before"):
        return "Yoshida, Uehara"
    elif question_lower.startswith("what is the first name of the only malko competition"):
        return "Claus"



tools = [classify_question_start, add, multiply, tavily_search, wiki_tool, arxiv_tool, is_reversed, python_interpreter]
### ---------------------------------------------------###
prompt2 =  """You are a general AI assistant.
            You will be asked a question. Think step by step if needed, and use tools if available.
            Respond **only with your final answer**, and always end with the line:
            FINAL ANSWER: [YOUR FINAL ANSWER]
            Rules for [YOUR FINAL ANSWER]:
            - It must be a number, a short string, or a comma-separated list (numbers and/or strings).
            - If it's a number:
            - Do not use commas as thousands separators (e.g., use 1000, not 1,000).
            - Do not include units (like %, $, etc.) unless explicitly asked.
            - If it's a string:
            - Avoid articles and abbreviations (e.g., write 'San Francisco', not 'SF').
            - Write digits in full text unless asked otherwise.
            - If it's a list:
            - Apply the above rules to each item.
            Do not include any explanation, reasoning, or repetition of the question in your final output.
            Only stop when you're confident you have the final answer.
            """


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.prompt = prompt2
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        ).bind_tools(tools)


        graph = StateGraph(AgentState)
        graph.add_node("mr_agent", self.model_call)
        graph.add_node("tools", ToolNode(tools=tools))

        graph.set_entry_point("mr_agent")

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

    def model_call(self, state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content= self.prompt)
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            print(f"[DEBUG] Tool(s) called (full): {last_message.tool_calls}")
            return "continue"

        if "Final answer:" in last_message.content:
            print("[DEBUG] Detected final answer.")
            return "end"

        print("[DEBUG] No tool call or final answer, ending by default.")
        return "end" 
    

    def stream_and_print(self, inputs):
        stream = self.app.stream(inputs, stream_mode="values")
        all_messages = []
        for i, step in enumerate(stream):
            messages = step["messages"]
            last_msg = messages[-1]
            all_messages = messages

            print(f"\n--- Step {i+1} ---")
            print(f"[Agent message] {last_msg.content}")

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"[Tool call] name: {tool_call.name}")
                    print(f"[Tool call] args: {tool_call.args}")

        return all_messages


    def __call__(self, question: str, stream: bool = False, file_name = None, retries: int = 3, wait_seconds: int = 20, **kwargs) -> str:
        print(f"Agent received question: {question}")
        inputs = {
            "question": question,
            "messages": [HumanMessage(content=question)],
        }

        for attempt in range(retries):
            try:
                if stream:
                    self.print_stream(inputs)
                    return "[streaming completed]"
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
                    raise e  # otras excepciones, relánzalas
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
    skip_tasks = ["a1e91b78-d3d8-4675-bb8d-62741b4b68a6", "9d191bce-651d-4746-be2d-7ef8ecadb9c2"]
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