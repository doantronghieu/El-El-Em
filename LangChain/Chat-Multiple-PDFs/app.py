import langchain

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain

from openai import embeddings

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ''

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # obj
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=1000, chunk_overlap=200,
        length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # option 1
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name='hkunlp/instructor-xl') # option 2
    vectorstore = faiss.FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI() # option 1
    # llm = HuggingFaceHub(
    #     repo_id='open-llm-leaderboard/details_cloudyu__Mixtral_34Bx2_MoE_60B')
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({
        'question': user_question
    })
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if (i % 2 == 0):
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs',
                       page_icon=':books:')
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
      st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your documents: ')
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader(
            'Upload your PDFs here and click on Process', accept_multiple_files=True)

        if st.button('Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)  # list
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

# streamlit run app.py
