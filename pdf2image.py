# STEP 1
# import libraries
import fitz
import os
import json
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Milvus
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from dotenv import load_dotenv
from glob import glob
load_dotenv()
# Constants for IBM COS values
COS_ENDPOINT = os.environ.get("COS_ENDPOINT") # Current list avaiable at https://control.cloud-object-storage.cloud.ibm.com/v2/endpoints
COS_API_KEY_ID = os.environ.get("COS_API_KEY_ID") # eg "W00YixxxxxxxxxxMB-odB-2ySfTrFBIQQWanc--P3byk"
COS_INSTANCE_CRN = os.environ.get("COS_INSTANCE_CRN")
bucket_name = "systex"
COS_OBJECT_URL = os.environ.get("COS_OBJECT_URL")
# Create client 
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

MILVUS_CONNECTION={"host": os.environ.get("MILVUS_HOST"), "port": os.environ.get("MILVUS_PORT")}
PDF_DIR = "/app/data/"
wx_model = os.getenv("WX_MODEL")
repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

hf = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)

# STEP 2
# file path you want to extract images from


def extract_text_image(file):

    IMAGE_DIR = file.split('/')[-1].split('.')[0]
    
    # open the file
    pdf_file = fitz.open(file)
    documents = []
    metadata_dict = {}
    # STEP 3
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):

        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images()
        image_sources = []
        # printing number of images found in this page
        
        if image_list:
            print(
                f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
            # metadata = ({'image_source': "", 'page':page_index+1})
            # documents.append(Document(page_content=page.get_text(),metadata=metadata))
        for image_index, img in enumerate(page.get_images(), start=1):

            # get the XREF of the image
            xref = img[0]

            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            image_src = os.path.join(IMAGE_DIR, f'{page_index}-{image_index}.{image_ext}')

            obj = cos.Object(bucket_name,  f'{image_src}')
            obj.put(Body=image_bytes)
            # with open(image_src , 'wb') as image_file:
            #     image_file.write(image_bytes)
            #     image_file.close()
            cos_url = COS_OBJECT_URL + image_src
            image_sources.append(cos_url)
        metadata = ({'image_source': json.dumps(image_sources,ensure_ascii=False), 'page':page_index+1})
        documents.append(Document(page_content=page.get_text().replace('\n',''),metadata=metadata))

    return documents
# hf2 = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

# index = Chroma.from_documents(
#         documents=documents,
#         embedding=hf,
#         collection_name=INDEX_NAME,
#         persist_directory=INDEX_NAME
#     )
for pdf in glob(PDF_DIR+"*.pdf"):
    collection_name = pdf.split('/')[-1].split('.')[0]
    documents = extract_text_image(file = pdf)

    index = Milvus.from_documents(
        collection_name=collection_name,
        documents=documents,
        embedding=hf,
        connection_args=MILVUS_CONNECTION
        )
    index = Milvus(
        collection_name=collection_name,
        embedding_function=hf,
        connection_args=MILVUS_CONNECTION
        )
    # index = Chroma(
    #         embedding_function=hf,
    #         collection_name=INDEX_NAME,
    #         persist_directory=INDEX_NAME
    #     )
    result = index.similarity_search("DQ2 模擬資料模式")
    print(result)