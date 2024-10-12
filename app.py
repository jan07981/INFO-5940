import streamlit as st
from openai import AzureOpenAI
from os import environ
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import PyPDF2

st.title("📄 Interactive File Q&A")

uploaded_docs = st.file_uploader("Upload your text or PDF documents", type=("txt", "pdf"), accept_multiple_files=True)

query = st.chat_input(
    "Ask a question based on the uploaded documents",
    disabled=not uploaded_docs,
)

# init language model
openai_llm = AzureChatOpenAI(
    azure_deployment="gpt-4-deployment",
    temperature=0.2,
    api_version="2023-06-01-preview",
)

# session state for file processing and msgs
if "conversation" not in st.session_state:
    st.session_state["conversation"] = [{"role": "assistant", "content": "Please upload documents and ask questions!"}]
if "file_data" not in st.session_state:
    st.session_state["file_data"] = {}
if "file_chunks" not in st.session_state:
    st.session_state["file_chunks"] = set()

for message in st.session_state.conversation:
    st.chat_message(message["role"]).write(message["content"])

if uploaded_docs:
    for file in uploaded_docs:
        if file.name not in st.session_state["file_data"]:
            try:
                # Read plain text files
                if file.type == "text/plain":
                    content = file.getvalue().decode("utf-8")
                # Process PDF files
                elif file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                else:
                    st.error(f"Unsupported file type: {file.name}. Only .txt or .pdf formats are allowed.")
                    continue
                
                doc = Document(page_content=content, metadata={"source": file.name})
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                doc_chunks = text_splitter.split_documents([doc])
                
                # saving chunks for retrieval
                st.session_state["file_data"][file.name] = "\n\n".join([chunk.page_content for chunk in doc_chunks])
                st.success(f"Processed {file.name}")
            except Exception as error:
                st.error(f"Error processing {file.name}: {error}")

template_str = """
You are an assistant providing concise answers. Use the context from the documents to answer the question briefly.
Question: {question}

Context: {context}

Answer:
"""
prompt_template = PromptTemplate.from_template(template_str)

if st.session_state["file_data"]:
    st.write("Documents ready for queries:")
    for doc_name in st.session_state["file_data"].keys():
        st.write(f"- {doc_name}")

    def retrieve_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retrieve_docs, "question": RunnablePassthrough()} |
        prompt_template |
        openai_llm |
        StrOutputParser()
    )

if query and uploaded_docs:
    azure_client = AzureOpenAI(
        api_key=environ['AZURE_OPENAI_API_KEY'],
        api_version="2023-03-15-preview",
        azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )

    st.session_state.conversation.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        combined_content = "\n\n".join([f"Content of {filename}:\n{content}" 
                                        for filename, content in st.session_state["file_data"].items()])
        
        stream = azure_client.chat.completions.create(
            model="gpt-4-deployment",
            messages=[
                {"role": "system", "content": f"Here's the file content:\n\n{combined_content}"},
                *st.session_state.conversation
            ],
            stream=True
        )
        assistant_response = st.write_stream(stream)

    st.session_state.conversation.append({"role": "assistant", "content": assistant_response})
