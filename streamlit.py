import os, dotenv
dotenv.load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
from tabula.io import read_pdf
from langchain_community.llms import Ollama
from langchain.chains.summarize import load_summarize_chain


llm = Ollama(model="gemma2:2b")
embeddings = OllamaEmbeddings(model="gemma2:2b")
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)



summarise_prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """
prompt_qa = PromptTemplate.from_template(

    """
    Answer the following question based only on the provided context:
    The context tells about the form 16 filled by the user and describes the amount he has entered in each category
    
    <context>
    {context}
    </context>
    
    question : {input}

    """
)
doc_chain = create_stuff_documents_chain(llm,prompt_qa)


def create_db_retriever(vectordb):
    retriever = vectordb.as_retriever()
    return retriever

def create_retriever_chain(retriever):
    retrieve_chain = create_retrieval_chain(retriever,doc_chain)
    return retrieve_chain


def text_creation(tab):
    text= []
    for df in tab:
        df = df.replace(np.nan, 0)
        result = ''
        column_names = df.columns.to_numpy()
        column_names = [str(element) for element in df if (element != 'Unnamed: 0' and element != 'Unnamed: 1' and element != 'Unnamed: 2' and element != 'Unnamed: 3')]
        res = ' is '.join(column_names)
        result +=res
        df = (df.to_numpy()).flatten()
        
        filtered_elements = [str(element) for element in df if element != 0]
        res = ' is '.join(filtered_elements)
        result+=res
        text.append(result)
    return text



def main():
    st.title("Taxation Chat LLM")
    st.write("Upload Part B of your Form 16")
    uploaded_file = st.file_uploader("Choose a (.pdf) file", type=["pdf"])

    if uploaded_file is not None:
        # try:
        tab = read_pdf(uploaded_file, pages='all')
        text = text_creation(tab)
        documents = [Document(page_content=i) for i in text]
        choice = st.radio(
            "What do you want to do?",
            ["Q/A", "Summarisation"],
            captions=[
                "Ask questions related to your taxation",
                "Summarise your taxation",
            ],
        )
        if choice == "Summarisation":
            prompt = PromptTemplate(template=summarise_prompt_template, input_variables=["text"])
            stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            try:
                st.write(stuff_chain.run(documents))
            except Exception as e:
                print(
                    
                    "The code failed since it won't be able to run inference on such a huge context and throws this exception: ",
                    e,
                )
        elif choice == "Q/A":
            docs =splitter.split_documents(documents)
            ques = st.text_input("Enter your question: ")
            vectordb = FAISS.from_documents(docs, embeddings)

            retriever = create_db_retriever(vectordb)
            retriever_chain = create_retriever_chain(retriever)
            resp = retriever_chain.invoke({"input":ques})
            st.write(resp['answer'])

                
                
        # except:
        #     st.write("Not the correct file")
        

















if __name__ == '__main__':
    main()


