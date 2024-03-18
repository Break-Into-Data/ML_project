import os

import dotenv
from langchain.globals import set_debug
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader


dotenv.load_dotenv()


DEBUG = True
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

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


def load_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    all_splits = text_splitter.split_documents(data)
    return all_splits


def init_vectorstore(data):
    return FAISS.from_documents(
        data,
        embedding=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY
        ),
    )


def init_llm(model):
    return ChatOpenAI(
        model_name=model,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )


def build_pipeline(llm, vectorstore, prompt):
    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )


def main():
    set_debug(DEBUG)
    
    url = "https://raw.githubusercontent.com/ollama/ollama/main/docs/faq.md"
    question = "Where are models stored on macos?"

    data = load_data(url)
    vectorstore = init_vectorstore(data)
    llm = init_llm("gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=PROMPT_TEXT
    )
    pipeline = build_pipeline(
        llm,
        vectorstore,
        prompt
    )

    result = pipeline.invoke({"query": question})
    print(result)


if __name__ == "__main__":
    main()
