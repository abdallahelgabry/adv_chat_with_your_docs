## Features

- **PDF without OCR**: Uses IBM Docling for structural chunking
- **PDF with OCR**: Converts scanned documents to images and uses a Vision LLM to accurately extract text.
- **DOCX without OCR**: Bypasses heavy libraries by extracting text directly from the raw XML of Microsoft Word files.
- **XLS without OCR**: Parses spreadsheets natively using Pandas.
- **URL without OCR**: Scrapes and chunks live webpages directly.


## Setup

1. **Configure your `.env` file:**
   Create a `.env` file in the root directory:
   ```env
   OCI_CONFIG_PATH="~/.oci/config"
   OCI_CONFIG_PROFILE="DEFAULT"
   OCI_COMPARTMENT_ID="your-compartment-ocid"
   OCI_SERVICE_ENDPOINT="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
   OCI_COHERE_MODEL_ID="cohere.command-a-03-2025"
   OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Run the app:**
   streamlit run main.py

