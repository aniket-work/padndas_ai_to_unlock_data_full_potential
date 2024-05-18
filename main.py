import os

from pandasai.llm.local_llm import LocalLLM
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai import Agent

load_dotenv()

# model = LocalLLM(
#    api_base="http://localhost:11434/v1",
#    model="llama3"
# )

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

st.title("Data analysis with PandasAI Agent")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    agent = Agent(data, config={"llm": model})
    prompt = st.text_input("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                response = agent.chat(prompt)
                print(response)
                if "temp_chart.png" in response:
                    st.image(response)
                else:
                    st.write(response)