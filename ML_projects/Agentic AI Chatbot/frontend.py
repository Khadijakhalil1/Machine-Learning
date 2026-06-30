#step 1: check if the file is uploaded


import streamlit as st

uploaded_file = st.file_uploader("Upload PDF",
                                  type="pdf",
                                  accept_multiple_files=False)


#step 2: Chatbot Skeleton (Question & Answer)

user_query = st.text_area("Enter Your prompt:", height=150, placeholder="Type your question here...")
