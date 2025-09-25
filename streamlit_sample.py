import streamlit as st
from openai import OpenAI

client = OpenAI() 

st.set_page_config(page_title="Hello Codespaces", layout="centered")

st.title("Welcome to INFO-5940 Fall 2025!")

st.title("👋 Hello from Codespaces!")
st.write("If you can see this page, Streamlit is running correctly inside your Codespace.")

# name = st.text_input("What is your name?")
# if name:
#     st.success(f"Nice to meet you, {name}!")

# Initialize the messages list
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    if msg["role"] == "system":
        st.chat_message(msg["role"].write(msg["content"]))


if prompt := st.text_input("please enter your content:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in client.chat.completions.create(
            model="openai.gpt-5",
            messages=st.session_state.messages,
            stream=True,
        ):
            chunk_message = getattr(chunk.choices[0].delta, "content", "")
            if chunk_message:
                full_response += chunk_message
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})



st.markdown("---")
st.caption("This is a test app for INFO 5940 Fall 2025.")