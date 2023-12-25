import os
import streamlit as st

import tempfile

from pipeline.base_pipeline import DocumentProcessor, DocumentQALLM

preprocessor = DocumentProcessor()

def get_extension(file_type):  
    """  
    get extension of uploaded file_type  
    """  
    mappings = {"application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}  
  
    file_extension = mappings.get(file_type, "")  
    return file_extension

st.title('Generic Vanilla RAG')

uploaded_files = st.file_uploader("Choose a file",
                                  type=['docx', 'pdf'],
                                  accept_multiple_files=True)

if uploaded_files:

    uploaded_file_names = []

    for uploaded_file in uploaded_files:

        if uploaded_file is not None:

            file_extension = get_extension(file_type=uploaded_file.type)
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tfile:   
                tfile.write(uploaded_file.getvalue())  
                st.write('Temporary file path:', tfile.name)  

            uploaded_file_names.append(tfile.name)

    st.write('Files Uploaded Successfully!')

    #process and chunk the documents
    parsed_documents = preprocessor.parse_documents(document_paths=uploaded_file_names)
    chunked_documents = preprocessor.chunk_documents(parsed_documents=parsed_documents, by_line=True)
    embeddings = preprocessor.embed_documents(chunked_documents=chunked_documents)

    #define a mini rag
    rag = DocumentQALLM(document_processor=preprocessor)

    user_input = st.text_input('query here:')
    if st.button('Ask your documents'):
        if user_input == '':
            st.write('Please enter a query.')
        else:
            # Process query and display result
            result = rag.query_with_llm(user_input)
            st.text_area('Chat:',
                         value=f'User query: {user_input}\nResponse: {result}',
                         height=200,
                         max_chars=None,
                         key=None)