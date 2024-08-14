import streamlit as st

# defines the prompts under predefined prompt buttons
def predefined_prompts(user_question,button_name):
    with st.sidebar:
        st.markdown("""
            <style>
                .sidebar-header {
                    color: white;
                    font-size: 24px; 
                    margin-bottom: 20px; 
                }
            </style>
        """, unsafe_allow_html=True)

        # Use a div with the class 'sidebar-header' for styling
        st.markdown('<div class="sidebar-header">Get Summary</div>', unsafe_allow_html=True)
        if st.button("Summarize Document", use_container_width=True):
            button_name = "Summarize Document"
            user_question="""
            Please provide a detailed summary of the content in the document. 
            Include the main topics, key points, and any significant details or findings mentioned in the document.
            Ensure that the summary is comprehensive and covers all major sections of the PDF. """

    return user_question,button_name