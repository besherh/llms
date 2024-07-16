import streamlit as st
from langchain.llms import Ollama
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Initialize the model
model = Ollama(model="llama3")

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Create the conversation chain
chain = ConversationChain(
    llm=model,
    memory=st.session_state.memory,
    verbose=True
)

st.title("Simple LLM Demo with Memory")

# Display conversation history
st.subheader("Conversation History")
for message in st.session_state.memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("ai").write(message.content)

# Get user input
input_text = st.chat_input("Enter your question here")

if input_text:
    # Display user message
    st.chat_message("human").write(input_text)
    
    # Generate response
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        response = chain.run(input_text)
        response_placeholder.write(response)

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.memory.clear()
    st.experimental_rerun()
