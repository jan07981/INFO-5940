import streamlit as st
from openai import OpenAI
from openai import AzureOpenAI
import PyPDF2
from os import environ
import langchain_openai
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.schema import Document

st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader(
    "Upload an article", 
    type=("txt", "pdf"), 
    accept_multiple_files=True
)

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

# init session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask questions about the article!"}]

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = {}

#display chat msgs
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_file:
    document = ""
    for file in uploaded_file:
        if file.name not in st.session_state["vectorstore"]:
            try:
                if file.type == "text/plain":
                    # read content from txt files
                    file_content = file.read().decode("utf-8")

                elif file.type == "application/pdf":
                    # reading text from pdf
                    pdf_reader = PyPDF2.PdfReader(file)
                    file_content = ""
                    for page in pdf_reader.pages:
                        file_content += page.extract_text()
                else:
                    st.error(f"This is an unsupported file type for {file.name}. Pdf and .txt formats supported only.")
                    continue
                        
                document = Document(page_content=file_content, metadata={"source": file.name})

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_text(document)
                
                st.session_state["vectorstore"][file.name] = "\n\n".join([chunk.page_content for chunk in chunks])
                
                st.success(f"Processed {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

if st.session_state["vectorstore"]:
    st.write("Processed files:")
    for file in st.session_state["vectorstore"].keys():
        st.write(f"- {file}")

if question and uploaded_file:
    client = AzureOpenAI(
        api_key=environ['AZURE_OPENAI_API_KEY'],
        api_version="2023-03-15-preview",
        azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )

    # user input uploaded to chat
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        all_contents = "\n\n".join([f"Content of {filename}:\n{content}" 
                                    for filename, content in st.session_state["vectorstore"].items()])
        
        stream = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": f"The file content:\n\n{all_contents}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    #show chat msg
    st.session_state.messages.append({"role": "assistant", "content": response})
