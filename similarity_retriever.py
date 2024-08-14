import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from prompts import *
import os
import pandas as pd
from datetime import datetime

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    retriever = vector_store.as_retriever(search_type="similarity", k=4)
    return retriever

def get_conversational_chain():
    prompt_template = """
    - Analyze the response text provided in the context.
    - Generate an answer in your own words, using the information from the context to craft a response that is clear and coherent.
    - Ensure that the response is not a direct copy from the response section, but rather a paraphrased version that conveys the same meaning.
    If answer is not found in context, please give answer- Answer is not available for this question
    
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    """
    model =ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain,prompt

def user_input(user_question, retriever, chain, prompt):
    retrieved_docs  = retriever.invoke(user_question)
    response = run_chain(chain, prompt, retrieved_docs, user_question)
    return response, retrieved_docs

def run_chain(chain, prompt, docs, user_question):
    response = chain.invoke(
        {"input_documents": docs, "question": user_question, "prompt": prompt},
        return_only_outputs=True)
    return response

def store_results(convo_uuid, query_id, path, method, result, docs):
    data = [{ "conversation_id": convo_uuid,
         "query_id": query_id,
         "test_time": datetime.now().isoformat(timespec="seconds"),
         "result": result["output_text"],
         "method":method,
         "source_documents": docs}]
    
    new_row_df = pd.DataFrame.from_dict(data)

    if not os.path.exists(path):
        new_row_df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
        pd.concat([df, new_row_df]).to_csv(path, index=False)

@st.cache_resource
def load_files_and_vectorstore(pdf):
    text = get_pdf_text(pdf)
    chunks = get_text_chunks(text)
    retriever = get_vector_store(chunks)
    chain, prompt = get_conversational_chain()
    return retriever, chain, prompt