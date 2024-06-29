# import module
import streamlit as st
import time
from gen_answer import generate_answer
from retrieval import get_relavance_passage
# Title
st.title("HỆ THỐNG HỎI ĐÁP HUST")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hãy nhập vào câu hỏi?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_res = ""
        holder = st.empty()
        # test
        relavance = get_relavance_passage(prompt, 5)
        answer = generate_answer(prompt, relavance)
        for word in answer.split():
            full_res += word + " "
            time.sleep(0.05)
            holder.markdown(full_res + "|")
        holder.markdown(full_res )
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":full_res
            }
        )
    # with st.chat_message("assistant"):
    #     st.markdown("Mimic: {}".format(prompt))