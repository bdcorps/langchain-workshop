from dotenv import dotenv_values, load_dotenv

load_dotenv()
config = dotenv_values(".env")

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import (SeleniumURLLoader, TextLoader,
                                        UnstructuredURLLoader)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Scrapes the text from the Wikipedia page
urls = ["https://en.wikipedia.org/wiki/Brad_Pitt"]
loader = SeleniumURLLoader(urls=urls)
data = loader.load()

# print (data)

# Splits the data into chunks of 1000 characters
llm = OpenAI(temperature=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Creates embeddings from each chunk
embeddings = OpenAIEmbeddings()

# Uploads them to Chroma Vector DB
vectordb = Chroma.from_documents(texts, embeddings)

# Creates a retrieval-based QA system
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())

# Runs a query
query = "When is Brad Pitt's birthday?"
a = qa.run(query)

print (a)
