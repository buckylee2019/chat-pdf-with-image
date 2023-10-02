import streamlit as st
from streamlit_chat import message
from genai.model import Model
from typing import Iterator
try:
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")
from langchain.embeddings import HuggingFaceHubEmbeddings
from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams, ReturnOptions
import os
from typing import List
import json
import os
from typing import Any, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from typing import Iterator
from pymilvus import utility
from pymilvus import connections
from langchain.vectorstores import Milvus
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# user_api_key = st.sidebar.text_input(
#     label="#### Your Bam API key 👇",
#     placeholder="Paste your Bam API key, pak-",
#     type="password")
user_api_key = os.environ.get("BAM_API_KEY")
system_prompt = st.sidebar.text_area(
    label="System prompt for model",
    placeholder= """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """
)


DEFAULT_SYSTEM_PROMPT = """\
    <s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. You should answer the question directly from the given documents, you are responsible for finding the best answer among all the documents. Follow the rules below:\
    Summarize the related documents to user question using the following format, Use Markdown to display : Topic of the document, Step by Step instruction for user question, Image sources from document\
    Display the list of Image sources of related document in the following format: ![image name](image source "IMAGE"),\
    If the question is not related to the given context, SAY You can't provide the specific answer.\
    Context:\
    \n\
    
"""

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.text += token
            self.container.markdown(self.text)

params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stream=True,
        top_k=50,
        top_p=1,
    )

WX_MODEL = os.environ.get("WX_MODEL")
creds = Credentials(user_api_key, "https://bam-api.res.ibm.com/v1")

repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MILVUS_CONNECTION={"host": os.environ.get("MILVUS_HOST"), "port": os.environ.get("MILVUS_PORT")}
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
connections.connect(
    host = os.environ.get("MILVUS_HOST"),
    port = os.environ.get("MILVUS_PORT")
)

collection_name = st.sidebar.selectbox("選擇欲查詢的產品",
        set(utility.list_collections()))


hf = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)

# search_params={
#                 "metric_type": "L2", 
#                 "offset": 5, 
#                 "ignore_growing": False, 
#                 "params": {"nprobe": 10}
#             },
vectorstore = Milvus(
    collection_name=collection_name,
    embedding_function=hf,
    connection_args=MILVUS_CONNECTION,
    
    )

def similarity_search(query: str):

    
    docs = vectorstore.similarity_search_with_score(query, k=3)
    context = '\n'.join([f"Document {idx+1}. {doc[0].page_content} Image Source: {doc[0].metadata['image_source']}" for idx,doc in enumerate(docs) if doc[1]>0.5])
    source = ""
    # source += '\n'.join([f'{idx+1}. {doc.metadata["local_name"]} in {doc.metadata["data_list_source_table"]}' for idx, doc in enumerate(docs)])
    
    return context, source
def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    # texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    history = ["Chat history:"]
    for user_input, response in chat_history:
        history.append(f'User: {user_input.strip()}\nAssistant: {response.strip()}')
    chathistory= "\n".join(history)
    texts  = f'{system_prompt}\nIf there is no result from knowledge base, you can leverage the chat history: {chathistory}<</SYS>>User question:{message.strip()}\nMarkdown:[/INST]'
    
    return  texts

clear_conversation = st.sidebar.button(
    label="Clear conversation"
)


if not system_prompt:
    system_prompt = DEFAULT_SYSTEM_PROMPT
if user_api_key:
    if "source" not in st.session_state:
        st.session_state.source = []

    def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str) -> Iterator[str]:
        result, source = similarity_search(message)
        system_prompt = system_prompt + "\nContext:\n"+ result
        prompt = get_prompt(message, chat_history, system_prompt)

        st.session_state.source.append(source)
        return prompt
        

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if clear_conversation:
        st.session_state.messages = []
        st.session_state.source = []
    for user_input,response in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

    if prompt := st.chat_input("想問什麼問題呢?"):
        # st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        new_prompt = run(message=prompt,chat_history=st.session_state.messages[-1:],system_prompt=system_prompt)
        # next(generator)
        
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
                callbacks=[stream_handler]
            )
            response = llm(new_prompt)


            message_placeholder.markdown(response)

        st.session_state.messages.append((prompt,response))
        