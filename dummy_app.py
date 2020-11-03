import json
from tempfile import NamedTemporaryFile
import streamlit as st
from .utils import json_loader

st.write("""
# Dummy app to get started
First goal is simply to plot a spectrum using **matchms**!
""")

uploaded_file = st.file_uploader("Choose a spectrum file...", type=['json', 'txt'])
#temp_file = NamedTemporaryFile(delete=False)
if uploaded_file is not None:
    #temp_file.write(uploaded_file.getvalue())
    if uploaded_file.name.endswith("json"):
        #spectrums = json.load(uploaded_file) 
        spectrums = json_loader(uploaded_file)
        st.write(spectrums[0].metadata)
        
        fig = spectrums[0].plot()
        st.pyplot(fig)
        