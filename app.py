import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Chat with Interview Blogs", page_icon="📖", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with Interview blogs")
st.info("You can view the blog here(https://medium.com/@pearlpullan10)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about interview experiences in the blogs!",
        }
    ]

#@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader('data')
    docs = reader.load_data()
    #print(docs)
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the interview experiences blogs and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the interview experiences written in the blogs. Keep 
        your answers based on 
        facts – do not hallucinate features. always look at everything in the data folder""",
    )
  
    index = VectorStoreIndex.from_documents(docs)

    #print('ref_docs ingested: ', len(index.ref_doc_info))
    #print('number of input documents: ', len(docs))  

    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)