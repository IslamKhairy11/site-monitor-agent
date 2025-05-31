# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:19:42 2025

@author: Eslam
"""
# -----------------------------------------------------------------------------
# pages/2_Chatbot.py - Chatbot Page using LLM (via llm_manager)
# -----------------------------------------------------------------------------

import streamlit as st
import time
# Import the LLM manager
from utils.llm_manager import get_llm_model, generate_text


# --- LLM Function using llm_manager ---
def get_chatbot_response(user_query, project_name, project_location, analysis_summary_text, chat_history, llm_type, api_key):
    """
    Generates a chatbot response using the selected LLM model via llm_manager.

    Args:
        user_query (str): The user's latest question.
        project_name (str): Current project name.
        project_location (str): Current project location.
        analysis_summary_text (str): Text summary from the site analysis page.
        chat_history (list): List of previous {"role": "user/assistant", "content": "..."} dicts.
        llm_type (str): The selected LLM type ("Gemini" or "OpenAI").
        api_key (str): The API key for the selected LLM.


    Returns:
        str: The chatbot's response.
    """
    print(f"Generating Chatbot Response using {llm_type}...")

    model_info, error = get_llm_model(llm_type, api_key)
    if model_info is None:
        return f"The chatbot is currently unavailable due to an LLM configuration issue: {error}"

    # Construct prompt with context
    context = f"You are a helpful assistant for a construction site monitoring application. " \
              f"The current project is '{project_name}' located at '{project_location}'. "
    if analysis_summary_text:
        context += f"The latest site analysis summary is: ```{analysis_summary_text}``` "
    else:
        context += "No site analysis has been performed yet. "

    # Prepare chat history for the LLM.
    # OpenAI API uses a specific message format. Gemini can often handle history in the prompt.
    # Let's format the history into the prompt for generality, but llm_manager.generate_text
    # can be updated to use the OpenAI chat history format if needed.
    history_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in chat_history])

    prompt = f"""{context}

Previous conversation:
{history_text}

User query: {user_query}
Assistant: """ # Prompt the model to start its response

    try:
        # Use the generic generate_text function from llm_manager
        # Pass history explicitly for OpenAI API, otherwise handle in prompt
        response = generate_text(model_info, prompt, llm_type, chat_history=chat_history if llm_type == "OpenAI" else None)
        return response
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

# --- Page Content ---
st.title("💬 Chatbot Assistant")

# Check for project context and LLM configuration
project_name = st.session_state.get('project_name', None)
project_location = st.session_state.get('project_location', None)
analysis_summary_text = st.session_state.get('analysis_summary_text', None) # Get text summary from session state
llm_type = st.session_state.get('selected_llm_type', None)
llm_api_key = st.session_state.get('llm_api_key', None)


if not st.session_state.get('project_id'): # Check for project ID specifically
    st.warning("⚠️ Please create or select a project on the main page first to provide context for the chat.")
    st.stop()

# Check LLM availability *after* project check
llm_available = llm_api_key is not None and llm_api_key != ""
if not llm_available:
    st.warning("⚠️ Please configure your LLM API key on the main page to use the chatbot.")
    st.stop()


st.info(f"Chatting about Project: **{project_name}**")
if analysis_summary_text:
    with st.expander("View Last Analysis Summary"):
        st.text(analysis_summary_text) # Use st.text or st.markdown for potentially long summaries
else:
    st.markdown("_No analysis performed yet for this project. Upload media on the 'Site Analysis' page for context._")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you with the project analysis?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the project or analysis..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass required info including LLM config
            response = get_chatbot_response(
                prompt,
                project_name,
                project_location,
                analysis_summary_text, # Pass the summary text
                st.session_state.messages[:-1], # Pass history excluding the latest user message for the prompt context
                llm_type,
                llm_api_key
            )
            st.markdown(response)
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})