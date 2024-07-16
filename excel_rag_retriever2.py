import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from langchain.llms import Ollama
from langchain.schema import AIMessage, HumanMessage, Document
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Initialize the model and embeddings
model = Ollama(model="llama3")
embeddings = HuggingFaceEmbeddings()

# Initialize memory and DataFrame
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "df" not in st.session_state:
    st.session_state.df = None

def process_dataframe(df):
    documents = []

    # Column names (stored separately for easy retrieval)
    column_info = f"""
    COLUMN_NAMES:
    Total Columns: {df.shape[1]}
    Column Names: {', '.join(df.columns.tolist())}
    """
    documents.append(Document(page_content=column_info, metadata={"type": "column_names"}))

    # Metadata
    metadata = f"""
    META_DATA:
    Total Rows: {df.shape[0]}
    Total Columns: {df.shape[1]}
    Data Types: {', '.join([f"{col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])}
    """
    documents.append(Document(page_content=metadata, metadata={"type": "metadata"}))

    # Sample data (first few rows)
    sample_data = f"""
    SAMPLE_DATA (First 5 rows):
    {df.head().to_string()}
    """
    documents.append(Document(page_content=sample_data, metadata={"type": "sample_data"}))

    # Full dataframe (for detailed queries)
    full_data = f"""
    FULL_DATA_FRAME:
    {df.to_string()}
    """
    documents.append(Document(page_content=full_data, metadata={"type": "full_data_frame"}))

    return documents

def get_retriever(documents):
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_qa_chain():
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Content: {page_content}"
    )
    document_variable_name = "context"
    llm_chain = LLMChain(
        llm=model,
        prompt=PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following information to answer the question. The information includes column names, metadata, sample data, and the full dataframe from the Excel file.

            For questions about column names or the structure of the dataframe, refer to the COLUMN_NAMES and META_DATA sections.
            For questions requiring data analysis, calculations, or chart creation, use the SAMPLE_DATA or FULL_DATA_FRAME sections and write Python code to perform the task.

            When writing Python code:
            1. Use the variable 'df' to refer to the dataframe.
            2. Use pandas, numpy, and matplotlib.pyplot for data manipulation and visualization.
            3. For charts, use plt.figure() to create a new figure, then use plt.savefig() to save it to a BytesIO object.
            4. Return the result or the BytesIO object containing the chart.

            Wrap your Python code in triple backticks like this:
            ```python
            # Your code here
            ```

            After the code block, explain what the code does and interpret the results.

            {context}

            Question: {question}
            Answer:"""
        )
    )
    return StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        document_prompt=document_prompt
    )

def execute_python_code(code, df):
    output = io.StringIO()
    safe_globals = {
        'df': df,
        'pd': pd,
        'np': np,
        'plt': plt,
        'io': io,
        'print': lambda *args: print(*args, file=output)
    }
    exec(code, safe_globals)
    chart_data = safe_globals.get('chart_data')
    return output.getvalue(), chart_data

# Streamlit UI
st.title("Improved Column Retrieval RAG Chatbot for Excel Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    st.session_state.df = pd.read_excel(uploaded_file)
    st.success("Excel file uploaded successfully!")

# Chat interface
st.subheader("Chat")
for message in st.session_state.memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("ai").write(message.content)

# User input
user_input = st.chat_input("Enter your question here")

if user_input and st.session_state.df is not None:
    st.chat_message("human").write(user_input)
    
    with st.chat_message("ai"):
        try:
            documents = process_dataframe(st.session_state.df)
            retriever = get_retriever(documents)
            qa_chain = get_qa_chain()
            
            retrieved_docs = retriever.get_relevant_documents(user_input)
            response = qa_chain.run(input_documents=retrieved_docs, question=user_input)
            
            st.write(response)
            
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
                st.code(code, language="python")
                
                output, chart_data = execute_python_code(code, st.session_state.df)
                
                if output:
                    st.text("Output:")
                    st.text(output)
                
                if chart_data:
                    st.image(chart_data)
            
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write(f"Debug info: {type(e)}, {e}")

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.df = None
    st.experimental_rerun()
