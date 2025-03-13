import streamlit as st
from streamlit_option_menu import option_menu


selected = option_menu(
    menu_title=None,
    options=["Home", "Projects", "Contact"],
    orientation="horizontal",
)

if selected == "Home":
    st.title(f"You have selected {selected}")
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")

