import streamlit as st
from uuid import uuid4
from prompts import *
from similarity_retriever import *
import os

st.set_page_config("Document Summarizer", layout='wide')
st.header("QnA with your PDF")

if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid4())

# Selection box for property selection
pdf_selection = st.selectbox("Select:", (os.listdir("PDFs")))
pdf = f"PDFs/{pdf_selection}"

# Loading all csv and pdf files from the specified property directory
retriever, chain, prompt = load_files_and_vectorstore(pdf)

# CSV file for logging each query and response details related to query
qna_tests_csv_path = "qna_tests.csv"
method = "Similarity Retriever"

# Custom CSS for sidebar and emoji colors
st.markdown("""
    <style>
        /* Sidebar styling */
        [data-testid=stSidebar] {
            background-color: #5072A7;
        }
        
        /* Chat message styling */
        .st-emotion-cache-1c7y2kd {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgba(255, 255, 255);
        }

        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgb(14, 17, 23);
        }

    </style>
""", unsafe_allow_html=True)

#### Display Chat history messages 
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialization
user_question = None
button_name = None

# Checking if the prompt given is predefined
user_question, button_name = predefined_prompts(user_question, button_name)

# Defining text box for ad hoc queries 
text_input_question = st.chat_input("Ask a Question")
if text_input_question:
    user_question = text_input_question
    button_name = user_question

st.session_state.current_prompt = user_question

if user_question:
    # Defines prompt inside predefined prompts; contains user query if provided
    st.session_state.current_prompt = user_question
    # Defines name of the predefined prompt; contains user query if provided
    st.session_state.current_button = button_name

    if "current_button" in st.session_state and "current_prompt" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": st.session_state.current_button})
        with st.chat_message("user"):
            st.write(st.session_state.current_button)

        if st.session_state.messages[-1]["role"] != "assistant":
            query_id = uuid4()  # Generate unique ID for each query  
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    # Call the RAG Chain
                    response, docs = user_input(st.session_state.current_prompt, retriever, chain, prompt)
                    st.write(response["output_text"])
            message = {"role": "assistant", "content": response["output_text"]}
            st.session_state.messages.append(message)
            
            # Storing query and response in qna_tests.csv
            store_results(st.session_state.session_uuid, query_id, qna_tests_csv_path, method, response, docs)
