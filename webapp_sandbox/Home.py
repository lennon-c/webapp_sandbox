import streamlit as st

def main():
    st.set_page_config(
            page_title="Home",
            page_icon="🏠",
        )

    st.title("Sandbox Using Streamlit") 
    # st.sidebar.title("Pages")
    st.markdown("Just a set of personal test exploring the Streamlit library.")
  
if __name__ == '__main__':
    main()


