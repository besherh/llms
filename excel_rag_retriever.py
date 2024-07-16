import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model and embeddings
model = Ollama(model="llama3")
embeddings = HuggingFaceEmbeddings()
#embeddings =  OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "df_info" not in st.session_state:
    st.session_state.df_info = None

# Function to process Excel file
# sample file: https://learn.microsoft.com/en-us/power-bi/create-reports/sample-financial-download
def process_excel(file):
    df = pd.read_excel(file)
    
    # Store DataFrame info
    st.session_state.df_info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "head": df.head().to_string(),
        "describe": df.describe().to_string()
    }
    
    # Convert DataFrame to text
    text = f"DataFrame Info:\n"
    text += f"Shape: {st.session_state.df_info['shape']}\n"
    text += f"Number of rows: {st.session_state.df_info['shape'][0]}\n"
    text += f"Number of columns: {st.session_state.df_info['shape'][1]}\n"
    text += f"Columns: {', '.join(st.session_state.df_info['columns'])}\n"
    text += f"Data Types:\n{pd.Series(st.session_state.df_info['dtypes']).to_string()}\n"
    text += f"\nFirst few rows:\n{st.session_state.df_info['head']}\n"
    text += f"\nSummary statistics:\n{st.session_state.df_info['describe']}\n"
    text += f"\nFull data:\n{df.to_string()}"
    with open("file.txt", "w") as file: 
        file.write(text)
    
    # Split text and create vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    st.session_state.vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Create QA chain with custom prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be precise and use exact numbers when available.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

# Streamlit UI
st.title("Improved RAG Chatbot with Excel Upload")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    process_excel(uploaded_file)
    st.success("Excel file processed and indexed!")

# Chat interface
st.subheader("Chat")
if st.session_state.memory.chat_memory.messages:
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
        try:
            response = st.session_state.qa_chain.run(user_input)
            if isinstance(response, dict):
                # If the response is a dict, extract the 'result' field
                response = response.get('result', str(response))
            elif not isinstance(response, str):
                # If the response is not a string, convert it to a string
                response = str(response)
            st.write(response)
            
            # Update memory
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(response)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            # Log the error for debugging
            st.write(f"Debug info: {type(response)}, {response}")

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.df_info = None
    st.experimental_rerun()
