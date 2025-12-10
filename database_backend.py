from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver  #<--
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3 #<--

# --- NEW IMPORTS FOR HUGGING FACE ---
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


#********************************************************************************
#  instead of the in memory saver we use "Sqlitesaver"
# we make a database connection ---> in checkpointer=sqlitesaver(conn=connection_name)

# we also add retrive_all_threads functions  to retrive threads from the database  
# which will be used by the ui part to display past chats 
#......................................................................
load_dotenv()



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

#database connection 
connection=sqlite3.connect(database='database.db',check_same_thread=False)


# Checkpointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):  #none because we wanrt to iterate for all checkpointers not any specific thread's checkpoints
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


