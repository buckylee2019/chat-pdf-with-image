# PDF Image Text Extraction and Similarity Search with watsonx

## Prerequisites

Before running the script, ensure you have the following prerequisites:

- Python 3.x installed on your system.
- Required Python libraries are installed. You can install them using `pip`:
- pip install -r requirements.txt


- Access to the following services and APIs:

  - IBM Cloud Object Storage (COS): You need an IBM COS instance and API key.
  - Hugging Face Hub: You need an API token for accessing Hugging Face models.
  - Milvus: You need the host and port information for Milvus.

- PDF documents from which you want to extract text and images.

## Installation

1. Clone the repository or download the script to your local machine.

2. Create a `.env` file in the same directory as the script and add the following environment variables with your own values:
```
COS_ENDPOINT=<IBM_COS_ENDPOINT>
COS_API_KEY_ID=<IBM_COS_API_KEY_ID>
COS_INSTANCE_CRN=<IBM_COS_INSTANCE_CRN>
COS_OBJECT_URL=<IBM_COS_OBJECT_URL>
MILVUS_HOST=<MILVUS_HOST>
MILVUS_PORT=<MILVUS_PORT>
HUGGINGFACEHUB_API_TOKEN=<HUGGINGFACEHUB_API_TOKEN>
WX_MODEL=<WX_MODEL>
```


   Replace `<IBM_COS_ENDPOINT>`, `<IBM_COS_API_KEY_ID>`, `<IBM_COS_INSTANCE_CRN>`, `<IBM_COS_OBJECT_URL>`, `<MILVUS_HOST>`, `<MILVUS_PORT>`, `<HUGGINGFACEHUB_API_TOKEN>`, and `<WX_MODEL>` with your actual credentials and values.

## Usage

To use the script, follow these steps:

1. Open a terminal and navigate to the directory containing the script and the `.env` file.

2. Run the script:
    
    pdf2image.py
   

3. The script will process the PDF document specified in the `file` variable and perform the following steps:

   - Extract text from each page of the PDF.
   - Extract images from each page of the PDF and upload them to IBM COS.
   - Store metadata containing image source URLs and page numbers for each page.
   - Test a similarity search using the specified text query.



## Script Overview

The script can be divided into the following major steps:

### Step 1: Importing Libraries and Setting Up Environment

- Imports necessary libraries for PDF processing, cloud services, and embeddings.
- Loads environment variables from the `.env` file, including IBM COS and Hugging Face credentials.

### Step 2: Configuration and Setup

- Specifies the file path of the PDF document to process.
- Sets up variables for COS bucket name, Milvus connection, and Hugging Face model information.

### Step 3: Text and Image Extraction

- Opens the PDF file and iterates through its pages.
- Extracts text from each page and stores it in a list of documents.
- Extracts images from each page, uploads them to IBM COS, and stores image source URLs in the document metadata.

## Running locally

If you plan to run on your local environment, 
1. You can modify the `.env_template` file and save as `.env`. 
2. ```docker build -t <image_name> .```
3. ```docker run -p 8501:8501 ``` <image_name>
4. Open browser and check localhost:8501

## Running on IBM Code Engine

1. Fork the repository
2. Under **Choose the code to run** section choose **Source code** and specify your git repo.
3. Under **Optional settings** section specify the environment variables list in `.env_template` file.


## Troubleshooting

If you encounter any issues or errors while running the script, consider the following:

- Verify that you have correctly set up the environment variables in the `.env` file with valid credentials and URLs.
- Ensure that the required Python libraries are installed.
- Check if the PDF file specified in the `file` variable exists and is accessible.

## License

This script is provided under the [MIT License](LICENSE). You are free to use, modify, and distribute it as needed.
