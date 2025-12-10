from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# --- NEW IMPORTS FOR HUGGING FACE ---
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()

# --- STEP 1: LOAD LOCAL MODEL ---
# Good small options: 
# 1. "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (Very fast, low memory)
# 2. "Qwen/Qwen2.5-0.5B-Instruct" (Newer, smarter, very small)
# 3. "microsoft/Phi-3-mini-4k-instruct" (Smarter, but requires more RAM)

model_id = "Qwen/Qwen2.5-0.5B-Instruct" 

print(f"Loading {model_id} locally... this may take a minute.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a standard Hugging Face pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.1,
    return_full_text=False,
    # device=0 # Uncomment this if you have a GPU (NVIDIA)
)

# Wrap it in LangChain's interface
hf_pipeline = HuggingFacePipeline(pipeline=pipe)
llm = ChatHuggingFace(llm=hf_pipeline)

# --- END NEW MODEL SETUP ---

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    # The local model works exactly the same way as the cloud one now
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

