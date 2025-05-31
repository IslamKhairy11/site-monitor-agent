# -----------------------------------------------------------------------------
# utils/llm_manager.py - Handles LLM Configuration and Text Generation
# -----------------------------------------------------------------------------

import google.generativeai as genai
import openai
import time
import streamlit as st # Use st.secrets if preferred, but direct input is requested

# Cache LLM models to avoid re-initializing on every rerun
@st.cache_resource
def get_llm_model(llm_type, api_key):
    """
    Initializes and returns the LLM model based on type and API key.
    Caches the model resource.
    """
    print(f"Attempting to initialize {llm_type} model...")
    if not api_key:
         print(f"Error: API key for {llm_type} is not provided.")
         return None, f"Error: API key for {llm_type} is not provided."

    try:
        if llm_type == "Gemini":
            genai.configure(api_key=api_key)
            # Using a common and capable Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            # Test connection/auth
            model.generate_content("hello", stream=False) # Simple test
            print("Gemini model initialized.")
            return model, None
        elif llm_type == "OpenAI":
            openai.api_key = api_key
            # Using a common and cost-effective OpenAI model
            model_name = "gpt-3.5-turbo"
            # Test connection/auth
            openai.chat.completions.create(model=model_name, messages=[{"role": "user", "content": "hello"}]) # Simple test
            print("OpenAI model initialized.")
            return model_name, None # Return model name for OpenAI API calls
        else:
            return None, f"Error: Unsupported LLM type '{llm_type}'"
    except Exception as e:
        print(f"Error configuring or connecting to {llm_type} API: {e}")
        return None, f"Error configuring or connecting to {llm_type} API: {e}"

def generate_text(model_info, prompt, llm_type, chat_history=None):
    """Generates text using the configured LLM."""
    if model_info is None:
        return "Error: LLM model not initialized."

    try:
        if llm_type == "Gemini":
             model = model_info
             # For Gemini, pass history as part of generate_content if needed, or handle in prompt
             # The current prompt structure includes history, so direct generate_content is fine.
             response = model.generate_content(prompt)
             return response.text
        elif llm_type == "OpenAI":
             model_name = model_info
             # OpenAI chat API uses a list of messages
             messages = [{"role": "user", "content": prompt}] # Start with current prompt
             if chat_history:
                 # Prepend history, ensuring roles are correct
                 # Assuming chat_history is a list of {"role": "user/assistant", "content": "..."}
                 messages = chat_history + messages

             response = openai.chat.completions.create(
                 model=model_name,
                 messages=messages
             )
             return response.choices[0].message.content
        else:
            return "Error: Unknown LLM type for text generation."
    except Exception as e:
        print(f"Error during {llm_type} text generation: {e}")
        return f"An error occurred during LLM text generation: {e}"

# Note: Reporting function will be in utils/reporting.py, but will call generate_text
# from this module. Chatbot logic will also call generate_text.