from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# --- GOOGLE IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
# --- LANGGRAPH IMPORTS ---
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. SETUP GOOGLE STACK
# -------------------

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# ---------------------------------------------------------
# 1. LLM: Qwen 2.5 72B (Running on Hugging Face Cloud)
# ---------------------------------------------------------
# We use 'HuggingFaceEndpoint' which connects to the free Inference API.
# This does NOT download the model to your laptop.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    # api_key is automatically read from os.environ["GROQ_API_KEY"]
)

# Wrap it in ChatHuggingFace to enable "Tool Binding" support
# llm = ChatHuggingFace(llm=llm_engine)

# ---------------------------------------------------------
# 2. EMBEDDINGS: MiniLM (Running on Hugging Face Cloud)
# ---------------------------------------------------------
# This sends text to HF API and gets vectors back. Zero local RAM usage.
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=os.environ["HF_TOKEN"],
#     model_name="Qwen/Qwen3-Embedding-8B" 
# )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# -------------------
# 2. Tools & Binding (CRITICAL STEP)
# -------------------
# You didn't include this in your snippet, but for the graph to work,
# you MUST tell the LLM about the tools using .bind_tools()

search_tool = DuckDuckGoSearchRun()
tools = [search_tool] 

# Qwen supports tool calling, but LlamaCpp integration can be tricky.
# This binds the tools so the model knows they exist.
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Ingest PDF and store retriever in global memory mapped to thread_id."""
    if not file_bytes: raise ValueError("No bytes received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # --- DEBUG: CHECK IF PDF WAS READ ---
        if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
            print("ERROR: PDF appears to be empty or contains only images (scanned).")
            return {"filename": filename, "chunks": 0, "error": "PDF is empty or scanned images."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # --- SAFETY CHECK: IF NO CHUNKS, STOP ---
        if not chunks:
            print("ERROR: Text splitting resulted in 0 chunks.")
            return {"filename": filename, "chunks": 0, "error": "No text chunks generated."}

        # Create Vector Store
        # Now safe to run because we know chunks exist
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        return {"filename": filename, "chunks": len(chunks)}
    finally:
        try: os.remove(temp_path)
        except: pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


# @tool
# def get_stock_price(symbol: str) -> dict:
#     """
#     Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
#     using Alpha Vantage with API key in the URL.
#     """
#     url = (
#         "https://www.alphavantage.co/query"
#         f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
#     )
#     r = requests.get(url)
#     return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# -------------------
# 1. LLM + embeddings
# -------------------

# # 1. Setup the GGUF Model (Replaces ChatOpenAI)
# model_id = "unsloth/Qwen3-4B-Instruct-2507-GGUF" 

# print(f"Loading {model_id} locally... this may take a minute.")

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Create a standard Hugging Face pipeline
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=256,
#     temperature=0.7,
#     repetition_penalty=1.1,
#     return_full_text=False,
#     # device=0 # Uncomment this if you have a GPU (NVIDIA)
# )

# # Wrap it in LangChain's interface
# hf_pipeline = HuggingFacePipeline(pipeline=pipe)
# llm = ChatHuggingFace(llm=hf_pipeline)

# 2. Setup Google EmbeddingGemma 300M (Replaces OpenAIEmbeddings)
# This downloads the model automatically from HuggingFace the first time.