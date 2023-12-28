from uuid import UUID
from langchain.schema.output import LLMResult
import streamlit as st
from streamlit_chat import message
from genai.model import Model
try:
    from langchain import PromptTemplate
    from langchain.chains import LLMChain,RetrievalQAWithSourcesChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")
from langchain.embeddings import HuggingFaceHubEmbeddings
from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams, ReturnOptions
import os
from typing import Any, List, Optional
import json
import os
from langchain.callbacks.base import BaseCallbackHandler
from pymilvus import utility
from pymilvus import connections
from langchain.vectorstores import Milvus
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Chat with Documents", page_icon="💡")
st.title("Systex RAG")

PREFIX_PROMPT = "<s>[INST] <<SYS>>"
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
You are an AI assistant tasked with providing answers by summarizing related documents or referring to previous chat history. You should follow these rules:
1. Summarize the content from the provided documents, using the following format:

Topic of the Document: Describe the topic of the document.
Step by Step Instruction: Provide user question-specific instructions or information from the document.
Image Sources from Document: If relevant, include image sources with Markdown format, like this: ![image text](image sources "IMAGE").

2. If the user's question is not related to the given context, check the chat history for relevant information. If no relevant information is found in the chat history, respond with "I can't answer the question."

By adhering to these rules, you will help users find accurate and valuable information.
    
"""

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.text += token
            self.container.markdown(self.text)
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    

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
use_history = st.sidebar.checkbox(label="use mermory")

hf = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)

vectorstore = Milvus(
    collection_name=collection_name,
    embedding_function=hf,
    connection_args=MILVUS_CONNECTION,
    
    )


def get_prompt_template(system_prompt=system_prompt, history=False):
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    document_with_metadata_prompt = PromptTemplate(
    input_variables=["page_content", "image_source"],
    template="\nDocument: {page_content}\n\tImage sources: {image_source}",
)
    if history:
        instruction = """
        Context: {history} \n {summaries}
        User question: {question}
        Answer the question in Markdown format,
        Markdown: """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "summaries", "question",], template=prompt_template)
    else:
        instruction = """
        Context: {summaries}
        User: {question}
        Answer the question in Markdown format,
        Markdown:
        """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["summaries", "question"], template=prompt_template)
    
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        document_with_metadata_prompt,
        prompt,
        memory,
    )

def retrieval_qa_pipline(db, use_history, llm, system_prompt):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - db (vectorestore): Specifies the preload vector db
    - system_prompt (str): Define from default or from web UI
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQAWithSourcesChain: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    retriever = db.as_retriever(search_kwargs={'k': 3})

    # get the prompt template and memory if set by the user.
    doc_promt, prompt, memory = get_prompt_template( system_prompt=system_prompt,history=use_history)

    # load the llm pipeline

    if use_history:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            chain_type_kwargs={"prompt":prompt,
                "document_prompt":doc_promt, "memory": memory},
        )
    else:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            chain_type_kwargs={
                "prompt":prompt,
                "document_prompt":doc_promt,
            },
        )

    return qa
clear_conversation = st.sidebar.button(
    label="Clear conversation"
)


if not system_prompt:
    system_prompt = DEFAULT_SYSTEM_PROMPT
if user_api_key:
    if "source" not in st.session_state:
        st.session_state.source = []


    if "messages" not in st.session_state:
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown("你好!\n我是精誠資訊股票分析系統專家小Q, 我能回答你操作股票系統的任何問題")

    if clear_conversation:
        st.session_state.messages = []
        st.session_state.source = []
    for user_input,response in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

    if prompt := st.chat_input("Ask your question.."):
        # st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)

            # message_placeholder            
            llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
                callbacks=[stream_handler]
            )
            qa_chain = retrieval_qa_pipline(vectorstore,True,llm,system_prompt)
            res = qa_chain(prompt,return_only_outputs=True)
            
        st.session_state.messages.append((prompt,res['answer']))
        