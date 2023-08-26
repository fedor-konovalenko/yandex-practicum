import streamlit as st
from model import temp_pred

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Metal Temperature prediction")

data = st.file_uploader("Choose a csv with features")

if st.button("Predict"):
    if data is not None:
        save_path = 'tmp/' + data.name
        with open(save_path, mode='wb') as w:
            w.write(data.getvalue())
        proc = save_path
        res, status = temp_pred(proc)

        st.write(status)
        if status == 'Succesfully predicted':
            st.write(res)

