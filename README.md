# INFO-5940
INFO 5940 Dev Env

## Overview

This RAG App is a web application built using Streamlit that allows users to upload text and PDF files, ask questions about their content, and receive responses powered by OpenAI's GPT-4 model. The application processes the uploaded files, splits their content into manageable chunks, and utilizes Azure OpenAI to provide answers based on the provided documents.

## Features

- Upload .txt and .pdf files for analysis.
- Ask questions related to the uploaded content.
- Efficient document processing using LangChain for text splitting.

## Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [OpenAI API](https://openai.com/api/)
- Azure OpenAI Service

## Installation

To set up and run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Add your keys for these: 
- AZURE_OPENAI_API_KEY=<your-api-key>
- AZURE_OPENAI_ENDPOINT=<your-endpoint>
- AZURE_OPENAI_MODEL_DEPLOYMENT=<your-model-deployment>

