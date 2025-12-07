import streamlit as st 
from langraph_backend_v1 import chatbot 
from langchain_core.messages import HumanMessage 

#for streaming we will use chatbot.stream instead of chatbot.invoke 
#streamlit--> here invui instead of st.chat_message we use st.stream 

config1={'configurable':{'thread_id':'t_11'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]


#loading the messages for ui 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input=st.chat_input('your query')

if user_input:

    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message=st.write_stream(
            message_chunk.content  for message_chunk ,metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},config=config1,
                stream_mode='messages'
            )
        )
    
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})


