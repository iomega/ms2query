import streamlit as st

st.write("""
# Dummy app to get started
First goal is simply to plot a spectrum using **matchms**!
""")

uploaded_file = st.file_uploader("Choose a file", type=['txt', 'jpg'])
if uploaded_file is not None:
    # do stuff
    pass
