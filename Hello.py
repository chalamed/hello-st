# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv


import os

os.environ['OPENAI_API_KEY'] = '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'


load_dotenv()
LOGGER = get_logger(__name__)

with st.sidebar:
    st.title("👀 L'art de rendre les documents vivants 👀")
    st.markdown('''
    ## About APP:

    Changeons nos abitudes

    ## About me:

    - [Linkedin](Chala Mohamed)

    ''')

    add_vertical_space(4)
    st.write("💬 Hier n'est plus, demain n'est pas encore. Nous n'avons qu'aujourd'hui. Commençons. 🤗")
    
def run():
    st.header("🧐 Faire parler les documents c'est possible 🧐")


    PINECONE_API_KEY = '580508ed-95a6-4500-844a-e07f37ca7657'
    PINECONE_ENV = 'gcp-starter'  # os.getenv('PINECONE_ENV')

    pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')
    index = pinecone.Index('indexrq')
    embeddings = OpenAIEmbeddings()
    vectordb = Pinecone.from_existing_index(index_name=index, embedding=embeddings, namespace=PINECONE_ENV)

    query = st.text_input("Ask questions about related your upload pdf file")
    # st.write(query)

    if query:
        # openai chat process
        #llm = OpenAI(temperature=0.8)
        #chain = load_qa_chain(llm=llm, chain_type="stuff")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        pdf_chat = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(),
                                                            memory=memory)

        with get_openai_callback() as cb:
            #response = chain.run(input_documents=docs, question=query)
            response = pdf_chat({"question": query})
            print(cb)
        st.write(response["answer"])


if __name__ == "__main__":
    run()
