# Stephan Raaijmakers, 2024
#!pip install -r requirements.txt
import os
import bs4
import dotenv
from operator import itemgetter
#from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
#from chromadb.utils import embedding_functions


# If you use COLAB: put your Huggingface/OpenAI keys in keys.env, or upload them in COLAB through the "key" icon.
# Otherwise put them in the code (be careful).
# Replace <keys.env> with your keys.env below:
# dotenv.load_dotenv('<keys.env>')

# OR USE (preferred, use key symbol in Colab):
#os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# It this throws an error, use os.environ["HUGGINGFACEHUB_API_TOKEN"] = <your key> (not recommended)

# For Dutch LLMs, see https://huggingface.co/spaces/BramVanroy/open_dutch_llm_leaderboard


llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature":0.1,"max_length":128})
# llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),model='gpt-3.5-turbo')
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":64})
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = HuggingFaceHub(repo_id="BramVanroy/Llama-2-13b-chat-dutch", model_kwargs={"temperature":0.1,"max_length":64})
# llm = HuggingFaceHub(repo_id="BramVanroy/GEITje-7B-ultra", model_kwargs={"temperature":0.1,"max_length":64})
# llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature":0.1, "max_length":64})

# Replace <question> with your question below:

#prompt = """Question: <question>?

#Let's think step by step.

#Answer: """

#print(llm(prompt))

# =============================== DOCUMENT LOADING ====================================

# Option 1: from the web
# Replace <url> with your url below:
loader = WebBaseLoader(
    web_paths=("<url>",),

    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
            #class_=("content_main")
        )
    ),
)
#docs = loader.load()

# Option 2: create a folder ./documents, with *.txt files.
docs=[]
n=0
for file in os.listdir("drive/MyDrive/rag_documents"):
  if file.endswith('.txt'):
    n+=1
    loader=TextLoader("drive/MyDrive/rag_documents/"+file)
    docs.extend(loader.load())

print("LOADED ",n, " documents")

# Process documents:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#embedding_function = SentenceTransformerEmbeddings(model_name="GroNLP/gpt2-medium-dutch-embeddings")

vectorstore = Chroma.from_documents(documents=splits,embedding=embedding_function)

# ========================== Retrieve and generate ==============================


retriever = vectorstore.as_retriever()

# Option 1:
#prompt = hub.pull("rlm/rag-prompt")

#def format_docs(docs):
#    return "\n\n".join(doc.page_content for doc in docs)


#rag_chain = (
#    {"context": retriever | format_docs, "question": RunnablePassthrough()}
#    | prompt
#    | llm
#    | StrOutputParser()
#)

# Replace <query> with your query below:
#print("\n================================================================\n")
#print("QUERY: <query>")
#print(rag_chain.invoke("<query>"))
#print("\n================================================================\n")

# Option 2:

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Replace <question> with your question below:
print(chain.invoke({"question": "<question>", "language": "dutch"}))







