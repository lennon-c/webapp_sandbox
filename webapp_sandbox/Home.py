import streamlit as st

def main():
    st.set_page_config(
            page_title="Home",
            page_icon="🏠",
        )

    st.title("Sandbox Using Streamlit") 
    # st.sidebar.title("Pages")
    st.markdown("Just a set of personal tests exploring the Streamlit library. ")

    """
    Code in my GitHub: [webapp_sandbox](https://github.com/lennon-c/webapp_sandbox)"""
  
if __name__ == '__main__':
    main()


