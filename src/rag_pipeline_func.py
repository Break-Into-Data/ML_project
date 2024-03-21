import os
from functools import cache

import dotenv
from faiss import IndexFlatL2
from langchain.globals import set_debug
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore


dotenv.load_dotenv()


DEBUG = os.getenv("DEBUG", "False").lower() == "true"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-3.5-turbo"

PROMPT_TEXT = """Human: 
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If the context does not contain the answer, just say that you don't know. 
-------- END PROMPT
Question: 
{question}
-------- END QUESTION
Context: 
{context}
-------- END CONTEXT
Answer:
"""

set_debug(DEBUG)


def _create_vectorstore():
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    dimensions: int = len(embedding_function.embed_query("dummy"))

    return FAISS(
        embedding_function=embedding_function,
        index=IndexFlatL2(dimensions),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )
    
    
@cache
def init():
    llm = ChatOpenAI(
        model_name=MODEL_NAME, 
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    prompt = PromptTemplate(
        input_variables=["question", "answer"], 
        template=PROMPT_TEXT
    )

    vectorstore = _create_vectorstore()
    pipeline = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    
    return pipeline, vectorstore


def scrape_url(vectorstore: FAISS, url: str):
    """ Scrapes a URL and stores the embeddings with the text 
        in the vectorstore 
    """
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    all_splits = text_splitter.split_documents(data)
    
    vectorstore.add_documents(all_splits)


def ask_question(pipeline, question: str):
    """ Calls the RAG pipeline to answer a question """
    result = pipeline.invoke({"query": question})
    return result["result"]


def reset_vectorstore(vectorstore: FAISS):
    """ Removes all documents from the vectorstore """
    ids_to_remove = []
    for doc_id in vectorstore:
        ids_to_remove.append(doc_id)
        
    vectorstore.delete(ids_to_remove)
