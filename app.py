#%%
import os
import glob
import gradio as gr
#%%
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

#%%
MODEL = "gpt-4o-mini"
db_name = "vector_db"

#%%
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

#%%

#%%
folder = glob.glob("Knowledge_base/*")
def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
if folder:
    for file in folder:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            resume_text = pages[0].page_content
            doc = Document(resume_text)
            doc = add_metadata(doc, "resume")
            documents.append(doc)
        else:
            
            loader = TextLoader(file, **text_loader_kwargs)
            text = loader.load()
            doc = text[0]
            doc = add_metadata(doc, "text")
            documents.append(doc)

print(len(documents))
#%%
from langchain_unstructured import UnstructuredLoader
page_url = "https://om-shewale.onrender.com/"
loader = UnstructuredLoader(web_url=page_url)
pages = []
async for page in loader.alazy_load():
    pages.append(page)
text = pages[0].page_content
doc = Document(text)
doc = add_metadata(doc, "text")
documents.append(doc)
    
    
    
    
#%%
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Total number of chunks: {len(chunks)}")
print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

#%%
embeddings = OpenAIEmbeddings()

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    

vector_db = Chroma.from_documents(documents= chunks, embedding=embeddings, persist_directory=db_name)

#%%
collection = vector_db._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)

#%%
llm = ChatOpenAI(temperature=0.5, model=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever= vector_db.as_retriever()
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

#%%
query = "What is Om currently studying?"
result = conversation_chain.invoke({"question": query})
#%%
def chatbot(question, history):
    result = conversation_chain.invoke({"question": question})
    return result['answer']
#%%
view = gr.ChatInterface(chatbot, type="messages").launch(inbrowser=True, share=True)
#%%
