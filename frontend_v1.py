import streamlit as st 
from langraph_backend_v1 import chatbot 
#chatbot object imported  from backend
from langchain_core.messages import HumanMessage 

#config id for the chat session 
config ={'configurable':{'thread_id':'t_101'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

#loading messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input=st.chat_input('your query here:')

if user_input:
    #load in history and then print 
    st.session_state['message_history'].append({'role':'user' ,'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response=chatbot.invoke({'messages':HumanMessage(content=user_input)},config=config)
    ai_message=response['messages'][-1].content

    #load ai in history and display 

    st.session_state['message_history'].append({'role':'assistant' ,'content':ai_message})
    with st.chat_message('assitant'):
        st.text(ai_message)




