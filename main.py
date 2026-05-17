import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import torchvision
except (ImportError, RuntimeError) as e:
    pass

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings

from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken
import streamlit as st
import os
import time
import tempfile
from PIL import Image
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()

OCI_CONFIG_PATH      = os.getenv("OCI_CONFIG_PATH")
OCI_CONFIG_PROFILE   = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
OCI_COMPARTMENT_ID   = os.getenv("OCI_COMPARTMENT_ID")
OCI_SERVICE_ENDPOINT = os.getenv("OCI_SERVICE_ENDPOINT")
OCI_COHERE_MODEL_ID  = os.getenv("OCI_COHERE_MODEL_ID", "cohere.command-a-03-2025")


if 'llm' not in st.session_state:
    st.session_state.llm = ChatOCIGenAI(
        model_id=OCI_COHERE_MODEL_ID,
        service_endpoint=OCI_SERVICE_ENDPOINT,
        compartment_id=OCI_COMPARTMENT_ID,
        auth_type="API_KEY",
        auth_profile=OCI_CONFIG_PROFILE,
        model_kwargs={"temperature": 0, "max_tokens": 4096},
        provider="cohere",
    )

if 'vision_llm' not in st.session_state:
    st.session_state.vision_llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_tokens=4096)

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint=OCI_SERVICE_ENDPOINT,
        compartment_id=OCI_COMPARTMENT_ID,
        auth_type="API_KEY",
        auth_profile=OCI_CONFIG_PROFILE,
    )

# initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'wind_logo' not in st.session_state:
    st.session_state.wind_logo = Image.open('wind logo.png')


custom_html = """
<div class="banner">
    <img src="https://wind-is.com/wp-content/uploads/2022/05/Original-Logo-01.png" width="100px" hieght="100px" alt="Banner Image">
</div>
<style>
    .banner { align: center; }
</style>
"""
st.components.v1.html(custom_html)
title = st.title("Let's Chat")

def render_messages():
    for message in st.session_state.chat_history:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=st.session_state.wind_logo):
                st.markdown(message["message"])
        else:
            with st.chat_message(message["role"], avatar=None):
                st.markdown(message["message"])

def toggleBtn():
    if 'uploaded_file' in st.session_state:
        del st.session_state['uploaded_file']
    if 'qa_chain' in st.session_state:
        del st.session_state['qa_chain']
    if 'retriever' in st.session_state:
        del st.session_state['retriever']
    if 'source_type' in st.session_state:
        del st.session_state['source_type']
    st.session_state.chat_history = []

on = st.toggle('General Talk', value=True, on_change=toggleBtn)

def create_hybrid_chunks(docling_result, source_name):
    
    # gpt tokinizer
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o"),
        max_tokens=128 * 1024,
    )
    
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=512,
        merge_peers=True,
        include_headings=True
    )
    
    chunk_iter = chunker.chunk(dl_doc=docling_result.document)
    
    all_splits = [
        Document(
            page_content=chunk.text,
            metadata={
                "source": source_name,
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunk_iter)
    ]
    
    return all_splits

def create_text_chunks(text_content, source_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    texts = text_splitter.split_text(text_content)
    all_splits = [Document(page_content=t, metadata={"source": source_name}) for t in texts]
    
    return all_splits

def build_vectorstore(chunks):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=st.session_state.embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 15})

def process_input_source():
    source_type = st.radio(
        "Choose input type:", 
        [
            "1 - PDF without OCR",
            "2 - PDF with OCR",
            "3 - DOCX without OCR",
            "4 - XLS without OCR",
            "5 - URL without OCR"
        ]
    )

    # option 1: PDF without OCR
    if source_type == "1 - PDF without OCR":
        file = st.file_uploader("Upload your PDF", type='pdf', key="uploaded_file_1")
        if file is not None:
            with st.status("Analyzing your PDF (without OCR)..."):
                try:
                    bytes_content = file.read()
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix=file.name, suffix='.pdf', delete=False
                    )
                    temp_file.write(bytes_content)
                    temp_file.close()

                    converter = DocumentConverter()

                    st.write("Processing PDF without OCR...")
                    result = converter.convert(temp_file.name)
                    
                    st.write(" PDF converted successfully")
                    
                    all_splits = create_hybrid_chunks(result, file.name)
                    
                    if not all_splits:
                        st.error("Could not create text chunks from the PDF.")
                        return False
                    
                    st.write(f" Created {len(all_splits)} semantic chunks for processing.")

                    # use OCI Cohere Embeddings
                    retriever = build_vectorstore(all_splits)
                    
                    # store retriever and source type in session state
                    st.session_state.retriever = retriever
                    st.session_state.source_type = "document"
                    
                    st.success("PDF processed successfully! You can now start chatting.")
                    
                    os.unlink(temp_file.name)
                    st.rerun()
                    return True
    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        return False

    # Option 2: PDF with OCR
    elif source_type == "2 - PDF with OCR":
        file = st.file_uploader("Upload your PDF", type='pdf', key="uploaded_file_2")
        if file is not None:
            with st.status("Analyzing your PDF with AI Vision OCR..."):
                try:
                    import base64
                    from pdf2image import convert_from_path
                    
                    bytes_content = file.read()
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix=file.name, suffix='.pdf', delete=False
                    )
                    temp_file.write(bytes_content)
                    temp_file.close()

                    st.write("Converting PDF pages to images...")
                    
                    # Convert pdf to img
                    try:
                        images = convert_from_path(temp_file.name, dpi=200)
                        st.write(f" Converted {len(images)} page(s) to images")
                    except Exception as pdf_error:
                        st.error(f"Failed to convert PDF to images: {str(pdf_error)}")
                        os.unlink(temp_file.name)
                        return False
                    
                    st.write("Extracting text using ocr...")
                    
                    vision_llm = st.session_state.vision_llm
                    all_extracted_text = []
                    
                    for page_num, image in enumerate(images, 1):
                        st.write(f"  Processing page {page_num}/{len(images)}...")
                        
                        # img to base64
                        import io
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Extract ALL text from this image. Include all text you see, including headers, body text, tables, captions, and any other visible text. Preserve the structure and formatting as much as possible. Return only the extracted text without any commentary."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        }
                                    }
                                ]
                            }
                        ]
                        
                        response = vision_llm.invoke(messages)
                        page_text = response.content
                        
                        all_extracted_text.append(f"\n\n--- Page {page_num} ---\n\n{page_text}")
                    
                    markdown_text = "\n".join(all_extracted_text)
                    
                    if not markdown_text or len(markdown_text.strip()) < 10:
                        st.error("No content could be extracted from the PDF.")
                        return False
                    
                    st.write(f" Extracted {len(markdown_text)} characters from the document.")
                    
                    all_splits = create_text_chunks(markdown_text, file.name)

                    if not all_splits:
                        st.error("Could not create text chunks from the PDF.")
                        return False
                    
                    st.write(f" Created {len(all_splits)} text chunks for processing.")
                    
                    # use oci cohere embeddings
                    retriever = build_vectorstore(all_splits)
                    
                    # retriever and source type storingg
                    st.session_state.retriever = retriever
                    st.session_state.source_type = "document"
                    
                    st.success("PDF processed successfully! You can now start chatting.")
                    
                    os.unlink(temp_file.name)
                    st.rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        return False
    
    # Option 3: DOCX without OCR
    elif source_type == "3 - DOCX without OCR":
        file = st.file_uploader("Upload your DOCX", type='docx', key="uploaded_file_3")
        if file is not None:
            with st.status("Analyzing your DOCX..."):
                try:
                    bytes_content = file.read()
                    temp_file = tempfile.NamedTemporaryFile(
                        prefix=file.name, suffix='.docx', delete=False
                    )
                    temp_file.write(bytes_content)
                    temp_file.close()

                    # extract text directly from docx XML
                    st.write("Processing DOCX by extracting XML content...")
                    
                    try:
                        import zipfile
                        from xml.etree import ElementTree as ET

                        extracted_text = []
                        
                        with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                            # read main document XML
                            xml_content = zip_ref.read('word/document.xml')
                            root = ET.fromstring(xml_content)
                            
                            # define namespaces for Word XML
                            namespaces = {
                                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                            }
                            
                            # Extract all text elements
                            for text_elem in root.findall('.//w:t', namespaces):
                                if text_elem.text:
                                    extracted_text.append(text_elem.text)
                    
                        markdown_text = " ".join(extracted_text)
                        st.write(" Successfully extracted DOCX XML content")
                        
                    except Exception as e:
                        st.write(f"XML extraction failed, trying Docling converter...")
                        converter = DocumentConverter()
                        result = converter.convert(temp_file.name)
                        markdown_text = result.document.export_to_markdown()
                    
                    if not markdown_text or len(markdown_text.strip()) < 10:
                        st.error("No content could be extracted from the DOCX.")
                        return False
                    
                    st.write(f" Extracted {len(markdown_text)} characters from the document.")
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=3000,
                        #chunk_overlap=200,
                        length_function=len,
                    )
                    
                    texts = text_splitter.split_text(markdown_text)
                    all_splits = [Document(page_content=t, metadata={"source": file.name}) for t in texts]
                    
                    if not all_splits:
                        st.error("Could not create text chunks from the DOCX.")
                        return False
                    
                    st.write(f" Created {len(all_splits)} text chunks for processing.")
                    
                
                    retriever = build_vectorstore(all_splits)
                    
                    st.session_state.retriever = retriever
                    st.session_state.source_type = "document"
                    
                    st.success("DOCX processed successfully! You can now start chatting.")
                    
                    os.unlink(temp_file.name)
                    st.rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"Error processing DOCX: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        return False

    # Option 4: XLS without OCR
    elif source_type == "4 - XLS without OCR":
        file = st.file_uploader("Upload your Excel file", type=['xls', 'xlsx'], key="uploaded_file_4")
        if file is not None:
            with st.status("Analyzing your Excel file..."):
                try:
                    import pandas as pd
                    
                    file_extension = os.path.splitext(file.name)[1]
                    
                    st.write("Processing Excel file...")
                    
                    # try docling first for better structure
                    use_docling = True
                    try:
                        bytes_content = file.read()
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix=file_extension,
                            delete=False
                        )
                        temp_file.write(bytes_content)
                        temp_file.close()
                        
                        converter = DocumentConverter()
                        result = converter.convert(temp_file.name)
                        
                        st.write(" Excel converted with Docling")
                        
                        all_splits = create_hybrid_chunks(result, file.name)
                        
                        if not all_splits or len(all_splits) == 0:
                            use_docling = False
                        else:
                            st.write(f" Created {len(all_splits)} semantic chunks")
                        
                        os.unlink(temp_file.name)
                        
                    except Exception as docling_error:
                        st.write(f"Docling approach failed, trying pandas...")
                        use_docling = False
                    
                    # if doc failed
                    if not use_docling:
                        excel_file = pd.ExcelFile(file)
                        sheet_names = excel_file.sheet_names
                        
                        st.write(f" Found {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
                        
                        all_content = []
                        
                        for sheet_name in sheet_names:
                            df = pd.read_excel(file, sheet_name=sheet_name)
                            
                            all_content.append(f"\n\n## Sheet: {sheet_name}\n")
                            
                            if not df.empty:
                                headers = " | ".join(str(col) for col in df.columns)
                                all_content.append(headers)
                                all_content.append("-" * len(headers))
                                
                                for idx, row in df.iterrows():
                                    row_text = " | ".join(str(val) for val in row.values)
                                    all_content.append(row_text)
                                
                                st.write(f"   Sheet '{sheet_name}': {len(df)} rows × {len(df.columns)} columns")
                            else:
                                all_content.append("(Empty sheet)")
                                st.write(f"  ⚠ Sheet '{sheet_name}' is empty")
                        
                        markdown_text = "\n".join(all_content)
                        st.write(f" Successfully extracted data using pandas")
                        

                        all_splits = create_text_chunks(markdown_text, file.name)

                    if not all_splits:
                        st.error("Could not create text chunks from the Excel file.")
                        return False
                    
                    st.write(f" Total: {len(all_splits)} chunks for processing.")
                    
                    # use oci Cohere Embeddings
                    retriever = build_vectorstore(all_splits)
                    
                    st.session_state.retriever = retriever
                    st.session_state.source_type = "document"
                    
                    st.success("Excel file processed successfully! You can now start chatting.")
                    
                    st.rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"Error processing Excel file: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        return False

    # Option 5: URL without OCR
    elif source_type == "5 - URL without OCR":
        url = st.text_input("Enter your document url:")
        if url and st.button("Load URL"):
            with st.status("Fetching content from the URL..."):
                try:
                    converter = DocumentConverter()
                    
                    st.write("Processing URL...")
                    result = converter.convert(url)
                    
                    st.write(" URL content fetched successfully")
                    
                    all_splits = create_hybrid_chunks(result, url)
                    
                    if not all_splits:
                        st.error("Could not create text chunks from the URL content.")
                        return False
                    
                    st.write(f" Created {len(all_splits)} semantic chunks for processing.")
                    
                    # use oci cohere embeddings
                    retriever = build_vectorstore(all_splits)
                    
                    # store retriever and source type in session state
                    st.session_state.retriever = retriever
                    st.session_state.source_type = "webpage"
                    
                    st.success("URL content loaded successfully! You can now start chatting.")
                    st.rerun()
                    return True
                except Exception as e:
                    st.error(f"Error loading the URL: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return False
        return False


def get_conversational_answer(user_input, retriever, source_type):
    """Uses OCI Cohere Command-A for document Q&A."""
    
    chat_history_str = ""
    for msg in st.session_state.chat_history[-6:]:
        role = "Human" if msg["role"] == "user" else "Assistant"
        chat_history_str += f"{role}: {msg['message']}\n"
    
    relevant_docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""You are a helpful and professional AI assistant designed to answer user questions based on {'uploaded documents' if source_type == 'document' else 'webpages'}.

Rules:
1. You can respond to greetings.
2. Use the conversation history to understand context and follow-up questions.
3. If the user refers to something from previous messages (like "it", "that", "the document", etc.), use the conversation history to understand what they're referring to.
4. If the user's question is not related to the provided {source_type}, reply: "I can't answer that question. I can only provide information based on the provided {source_type}".
5. Always respond in the same language as the user's question (English or Arabic).
6. Be concise but thorough in your answers.
7. If the response contains values (e.g., nan, null, none), don't include them in the final answer.

Conversation History:
{chat_history_str}

Context from {source_type}:
{context}

Current Question: {user_input}

Helpful Answer:"""
    
    response = st.session_state.llm.invoke(prompt)
    return response.content


if not on:
    title.title("chat with document or url")

    if 'retriever' not in st.session_state:
        processed = process_input_source()
        if not processed:
            st.write("Please choose an option and upload/enter content to start.")
    else:
        if user_input := st.chat_input("You:", key="user_input"):
            render_messages()
            with st.chat_message("user", avatar=None):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar=st.session_state.wind_logo):
                with st.spinner("Assistant is typing..."):
                    response_text = get_conversational_answer(
                        user_input, 
                        st.session_state.retriever, 
                        st.session_state.source_type
                    )

                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_text:
                    full_response += chunk
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "user", "message": user_input})
            st.session_state.chat_history.append({"role": "assistant", "message": response_text})
            st.rerun()
        else:
            render_messages()

else:
    title.title("Let's Chat")

    if user_input := st.chat_input("You:", key="user_input"):
        render_messages()

        with st.chat_message("user", avatar=None):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=st.session_state.wind_logo):
            with st.spinner("Assistant is typing..."):
                # General Talk uses OCI Cohere Command-A
                prompt_template = """
                You are a knowledgeable chatbot, here to help with questions from the user.
                Your tone should be professional and informative.
                Keep the answer concise and helpful.

                Chat History: {chat_history}

                User: {input}
                Assistant:
                """

                prompt = PromptTemplate(template=prompt_template, input_variables=["input", "chat_history"])
                chat_history_str = "\n".join([f"{m['role']}: {m['message']}" for m in st.session_state.chat_history])
                input_with_history = prompt.format(input=user_input, chat_history=chat_history_str)
                response = st.session_state.llm.invoke(input_with_history)

            message_placeholder = st.empty()
            full_response = ""
            for chunk in response.content:
                full_response += chunk
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "assistant", "message": response.content})
        st.rerun()
    else:
        render_messages()