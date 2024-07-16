import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize the model and embeddings
model = Ollama(model="llama2")
embeddings = HuggingFaceEmbeddings()

# Initialize memory and vector store
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Function to process Excel file
def process_excel(file):
    df = pd.read_excel(file)
    text = df.to_string()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    st.session_state.vectorstore = FAISS.from_texts(texts, embeddings)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever()
    )

# Streamlit UI
st.title("Basic RAG Chatbot with Excel Upload")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    process_excel(uploaded_file)
    st.success("Excel file processed and indexed!")

# Chat interface
st.subheader("Chat")
for message in st.session_state.memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("ai").write(message.content)

# User input
user_input = st.chat_input("Enter your question here")

if user_input and st.session_state.qa_chain:
    st.chat_message("human").write(user_input)
    
    with st.chat_message("ai"):
        response = st.session_state.qa_chain.run(user_input)
        st.write(response)
    
    # Update memory
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(response)

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.memory.clear()
    st.experimental_rerun()

Version 4 of 4
