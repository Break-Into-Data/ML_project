from faiss import IndexFlatL2
from langchain.globals import set_debug
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore

DEBUG = True
set_debug(DEBUG)

PROMPT_TEXT = """Human: 
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the context does not contain the answer, just say that you don't know. 
-------- END PROMPT
Question: 
{question}
-------- END QUESTION
Context: 
{context}
-------- END CONTEXT
Answer:
"""


def create_vectorstore():
    embedding_function = OpenAIEmbeddings()
    dimensions: int = len(embedding_function.embed_query("dummy"))

    return FAISS(
        embedding_function=embedding_function,
        index=IndexFlatL2(dimensions),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )


class RAGPipeline:
    model_temperature = 0
    model_name = "gpt-3.5-turbo"
    
    def __init__(self) -> None:
        llm = ChatOpenAI(
            model_name=self.model_name, 
            temperature=self.model_temperature,
        )
        prompt = PromptTemplate(
            input_variables=["question", "answer"], 
            template=PROMPT_TEXT
        )
        
        self.vectorstore = create_vectorstore()
        self.pipeline = RetrievalQA.from_chain_type(
            llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

    def scrape_url(self, url: str):
        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100,
        )
        all_splits = text_splitter.split_documents(data)
        
        self.vectorstore.add_documents(all_splits)

    def ask_question(self, question: str):
        result = self.pipeline.invoke({"query": question})
        return result["result"]

    def reset_vectorstore(self):
        """ Remove all documents from the vectorstore """
        ids_to_remove = []
        for doc_id in self.vectorstore:
            ids_to_remove.append(doc_id)
            
        self.vectorstore.delete(ids_to_remove)
