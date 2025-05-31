# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:17:50 2025

@author: Eslam
"""

# -----------------------------------------------------------------------------
# app.py - Main Page for the Construction Site Monitor AI Agent
# Handles Project Management and LLM Configuration.
# -----------------------------------------------------------------------------

import streamlit as st
import os
import sys
import shutil # For deleting project directories

# Add utils directory to the Python path
utils_path = os.path.join(os.getcwd(), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Check if utils directory exists, provide guidance if not
if not os.path.isdir("utils"):
    st.error("Error: The 'utils' directory containing analysis, reporting, database, and llm logic is missing.")
    st.stop()

try:
    from utils import analysis, reporting, database, llm_manager # Try importing all utils
    database.init_db() # Initialize database on app startup
except ImportError as e:
     st.error(f"Error: Could not import modules from the 'utils' directory. Ensure it has '__init__.py' and necessary files (analysis.py, reporting.py, database.py, llm_manager.py). Error: {e}")
     st.stop()
except Exception as e:
     st.error(f"Error during utils initialization or database setup: {e}")
     st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Site Monitor AI Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
def load_css(file_name):
    """Loads and injects CSS from a file located relative to the script."""
    try:
        # Get the directory of the current script (Main_Page.py)
        script_dir = os.path.dirname(__file__)
        css_path = os.path.join(script_dir, file_name) # Construct the full path

        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Use st.sidebar.warning if you prefer it less intrusive on the main page
        st.sidebar.warning(f"CSS file '{file_name}' not found in {script_dir}.")
    except Exception as e:
        st.sidebar.warning(f"Could not load CSS: {e}")

# Load CSS from the root directory (assuming .style.css is in the same dir as Main_Page.py)
load_css(".style.css")


# --- Initialize Session State ---
# Use session state to store information across pages
if 'project_id' not in st.session_state:
    st.session_state.project_id = None # ID from DB
if 'project_name' not in st.session_state:
    st.session_state.project_name = None
if 'project_location' not in st.session_state:
    st.session_state.project_location = None

# Analysis results (cleared when project changes or new file uploaded)
if 'analysis_summary_text' not in st.session_state:
    st.session_state.analysis_summary_text = None # Raw summary text from analysis
if 'safety_violations' not in st.session_state:
    st.session_state.safety_violations = None # Structured list of violations
if 'processed_media_path' not in st.session_state:
    st.session_state.processed_media_path = None # Path to saved annotated image/video
if 'processed_media_type' not in st.session_state:
    st.session_state.processed_media_type = None # 'image_annotated', 'video_annotated'
# Store the original filename for download purposes
if 'original_uploaded_filename' not in st.session_state:
     st.session_state.original_uploaded_filename = None


# Reporting results (cleared when analysis changes or new file uploaded)
if 'llm_generated_report' not in st.session_state:
    # Store the structured result from LLM generation (e.g., findings, recommendations)
    st.session_state.llm_generated_report = None
if 'pdf_preview_path' not in st.session_state:
    st.session_state.pdf_preview_path = None # Path to the last generated temporary PDF for preview

# LLM Configuration
if 'selected_llm_type' not in st.session_state:
    st.session_state.selected_llm_type = "Gemini" # Default LLM
if 'llm_api_key' not in st.session_state:
    st.session_state.llm_api_key = "" # User input API key

# PDF Customization (default values)
if 'pdf_body_font_size' not in st.session_state:
    st.session_state.pdf_body_font_size = 10
if 'pdf_heading1_font_size' not in st.session_state:
    st.session_state.pdf_heading1_font_size = 18
if 'pdf_heading2_font_size' not in st.session_state:
    st.session_state.pdf_heading2_font_size = 14
if 'pdf_image_width_inch' not in st.session_state:
     st.session_state.pdf_image_width_inch = 6 # Default image width in inches


# --- Main Page Content ---
st.title("🏗️ Eagle.ai | Construction Site Monitor AI Agent")
st.subheader("Your intelligent assistant for site analysis and monitoring.")

st.markdown(
    """
    Welcome to the Site Monitor AI Agent! This tool helps you analyze construction site
    images and videos using Roboflow models to identify key elements, ensure safety compliance,
    and generate summary reports.

    **Use Cases:**
    *   **Safety Compliance:** Detect Personal Protective Equipment (PPE) usage.
    *   **Progress Monitoring:** Identify structural elements and track changes.
    *   **Equipment Tracking:** Monitor the presence and location of heavy machinery like excavators.
    *   **Automated Reporting:** Generate summaries and reports based on visual data.

    **How it Works:**
    1.  **Set up LLM:** Configure your chosen Large Language Model API key below.
    2.  **Manage Projects:** Create a new project or select an existing one below.
    3.  **Analyze Site Media:** Upload images or videos on the 'Site Analysis' page.
    4.  **Review Results & Report:** View analysis results, customize, and generate a PDF report on the 'Site Analysis' page.
    5.  **Ask Questions:** Use the 'Chatbot' page for project-specific queries.

    Navigate using the sidebar on the left.
    """
)
st.divider()

# --- LLM Configuration Section ---
st.header("LLM Configuration")
st.info("Configure your Large Language Model API key. This is required for generating reports and using the chatbot.")

col1_llm, col2_llm = st.columns(2)

with col1_llm:
    selected_llm_type = st.selectbox(
        "Select LLM Provider",
        ["Gemini", "OpenAI"], # Add other providers here if supported
        index=["Gemini", "OpenAI"].index(st.session_state.selected_llm_type) if st.session_state.selected_llm_type in ["Gemini", "OpenAI"] else 0, # Handle potential invalid state
        key="llm_type_select"
    )

with col2_llm:
    llm_api_key = st.text_input(
        f"Enter {selected_llm_type} API Key",
        value=st.session_state.llm_api_key,
        type="password", # Hide the key
        key="llm_api_key_input"
    )

# Update session state with selected LLM and key
st.session_state.selected_llm_type = selected_llm_type
st.session_state.llm_api_key = llm_api_key

# Optional: Test LLM connection
if st.button("Test LLM Connection"):
    if not llm_api_key:
        st.warning("Please enter an API key first.")
    else:
        with st.spinner(f"Testing connection to {selected_llm_type}..."):
            model_info, error = llm_manager.get_llm_model(selected_llm_type, llm_api_key)
            if model_info:
                st.success(f"Successfully connected to {selected_llm_type}!")
            else:
                st.error(f"Failed to connect to {selected_llm_type}: {error}")


st.divider()

def initiate_delete_project(project_id):
    """Sets session state to initiate project deletion confirmation."""
    st.session_state.delete_confirmation_pending = True
    st.session_state.delete_confirmation_project_id = project_id
    # No rerun needed here, the state change triggers it


def cancel_delete_project():
    """Cancels the project deletion confirmation."""
    st.session_state.delete_confirmation_pending = False
    st.session_state.pop('delete_confirmation_project_id', None) # Clear the stored ID
    # No rerun needed here


def perform_delete_project(project_id):
    """Performs the actual database and file system deletion."""
    with st.spinner(f"Deleting project ID {project_id}..."):
        if database.delete_project(project_id):
            st.success(f"Project deleted.") # Generic message, project name might be gone from state
            # Clear session state for the deleted project
            st.session_state.project_id = None
            st.session_state.project_name = None
            st.session_state.project_location = None
            st.session_state.analysis_summary_text = None
            st.session_state.safety_violations = None
            st.session_state.processed_media_path = None
            st.session_state.processed_media_type = None
            st.session_state.original_uploaded_filename = None
            st.session_state.llm_generated_report = None
            st.session_state.pdf_preview_path = None
            # Clear confirmation state
            st.session_state.delete_confirmation_pending = False
            st.session_state.pop('delete_confirmation_project_id', None)
            st.rerun() # Rerun to update project list and clear details
        else:
            st.error(f"Failed to delete project ID {project_id}.")
            # Clear confirmation state even on failure
            st.session_state.delete_confirmation_pending = False
            st.session_state.pop('delete_confirmation_project_id', None)
            # No rerun needed here unless you want to refresh the project list

# Initialize new session state variables for deletion confirmation
if 'delete_confirmation_pending' not in st.session_state:
    st.session_state.delete_confirmation_pending = False
if 'delete_confirmation_project_id' not in st.session_state:
     st.session_state.delete_confirmation_project_id = None

# --- Project Management Section ---
st.header("Project Management")

# --- Select Existing Project ---
projects = database.get_projects()
project_options = {p['name']: p['id'] for p in projects}

# Calculate the complete list of options for the selectbox
options_list = ["-- Select Project --"] + sorted(list(project_options.keys()))

# Determine the name of the currently selected project from session state
# Default to "-- Select Project --" if no project is selected or the ID in state is invalid
current_project_data = database.get_project(st.session_state.project_id) if st.session_state.project_id else None
current_project_name_display = current_project_data['name'] if current_project_data else "-- Select Project --"

# Find the index of the current project name within the options_list
# Use a try-except block in case the current_project_name_display is somehow not in the list
try:
    initial_index = options_list.index(current_project_name_display)
except ValueError:
    # If the current project name is not found (e.g., project was deleted externally),
    # default the index to the first item ("-- Select Project --")
    initial_index = 0

# Now use the calculated options_list and initial_index in st.selectbox
selected_project_name = st.selectbox(
    "Select an existing project",
    options=options_list, # Use the correctly assembled list
    index=initial_index, # Use the calculated integer index
    key="project_select"
)

# Load selected project into session state if it changed
if selected_project_name != "-- Select Project --":
    selected_project_id = project_options.get(selected_project_name) # Use .get for safety
    # Only update state if a valid project was selected AND it's different from current
    if selected_project_id is not None and st.session_state.project_id != selected_project_id:
        # Project changed, update session state and clear previous analysis/report
        project_data = database.get_project(selected_project_id)
        if project_data:
            st.session_state.project_id = project_data['id']
            st.session_state.project_name = project_data['name']
            st.session_state.project_location = project_data['location']
            # Clear analysis and reporting state when project changes
            st.session_state.analysis_summary_text = None
            st.session_state.safety_violations = None
            st.session_state.processed_media_path = None
            st.session_state.processed_media_type = None
            st.session_state.original_uploaded_filename = None
            st.session_state.llm_generated_report = None
            st.session_state.pdf_preview_path = None
            st.success(f"Project '{st.session_state.project_name}' loaded.")
            st.rerun() # Rerun to update select box and sidebar
        else:
            st.error("Failed to load project details.")
            # Reset project state if loading failed
            st.session_state.project_id = None
            st.session_state.project_name = None
            st.session_state.project_location = None
            st.session_state.analysis_summary_text = None # Clear associated data
            st.session_state.safety_violations = None
            st.session_state.processed_media_path = None
            st.session_state.processed_media_type = None
            st.session_state.original_uploaded_filename = None
            st.session_state.llm_generated_report = None
            st.session_state.pdf_preview_path = None

elif selected_project_name == "-- Select Project --" and st.session_state.project_id is not None:
     # If "-- Select Project --" is selected, clear the current project state
     st.session_state.project_id = None
     st.session_state.project_name = None
     st.session_state.project_location = None
     st.session_state.analysis_summary_text = None # Clear associated data
     st.session_state.safety_violations = None
     st.session_state.processed_media_path = None
     st.session_state.processed_media_type = None
     st.session_state.original_uploaded_filename = None
     st.session_state.llm_generated_report = None
     st.session_state.pdf_preview_path = None
     st.rerun() # Rerun to update sidebar and analysis page checks

# --- Create New Project ---
st.subheader("Create New Project")
with st.form("new_project_form"):
    new_project_name = st.text_input(
        "New Project Name",
        placeholder="e.g., Coastal Bridge Repair"
    )
    new_project_location = st.text_input(
        "New Project Location",
        placeholder="e.g., Highway 1, Mile Marker 50"
    )
    create_submitted = st.form_submit_button("Create Project")

    if create_submitted:
        if not new_project_name or not new_project_location:
            st.warning("Please provide both Project Name and Location for the new project.")
        else:
            with st.spinner("Creating new project..."):
                # database.create_project returns the new ID or None if creation failed (e.g., duplicate name)
                new_id = database.create_project(new_project_name, new_project_location)
                if new_id:
                    st.session_state.project_id = new_id
                    st.session_state.project_name = new_project_name
                    st.session_state.project_location = new_project_location
                     # Clear analysis and reporting state for the new project
                    st.session_state.analysis_summary_text = None
                    st.session_state.safety_violations = None
                    st.session_state.processed_media_path = None
                    st.session_state.processed_media_type = None
                    st.session_state.original_uploaded_filename = None
                    st.session_state.llm_generated_report = None
                    st.session_state.pdf_preview_path = None
                    st.success(f"New project '{st.session_state.project_name}' created and selected!")
                    st.rerun() # Rerun to update select box and sidebar
                else:
                    # This 'else' block is reached if database.create_project returned None.
                    # Based on database.py's logic, this is specifically when a project
                    # with that name already exists (IntegrityError).
                    st.error(f"A project named '{new_project_name}' already exists. Please choose a different name or select the existing project from the dropdown above.")
                #else:
                #    st.error(f"Could not create project '{new_project_name}'. It might already exist.")

# --- Display Current Project & Delete Option ---
st.subheader("Current Project Details")
if st.session_state.project_id:
    st.write(f"**ID:** {st.session_state.project_id}")
    st.write(f"**Name:** {st.session_state.project_name}")
    st.write(f"**Location:** {st.session_state.project_location}")

    # Confirmation handling for delete button
    delete_confirm = st.button("Delete Current Project", key="delete_project_btn", on_click=initiate_delete_project,args=(st.session_state.project_id,))
    if delete_confirm:
         # Use a confirmation state
         st.session_state.confirm_delete_project = st.session_state.project_id
         st.warning(f"Are you absolutely sure you want to delete project '{st.session_state.project_name}' (ID: {st.session_state.project_id})? This action is permanent and will delete all associated files.")
         st.button("Cancel Deletion", key="cancel_delete_btn", on_click=lambda: st.session_state.pop('confirm_delete_project', None))
         confirm_now = st.button("Yes, Delete Project Permanently", key="confirm_delete_now_btn", on_click=initiate_delete_project,args=(st.session_state.project_id,))

         if confirm_now and 'confirm_delete_project' in st.session_state and st.session_state.confirm_delete_project == st.session_state.project_id:
            with st.spinner(f"Deleting project {st.session_state.project_name}..."):
                if database.delete_project(st.session_state.project_id):
                    st.success(f"Project '{st.session_state.project_name}' deleted.")
                    # Clear session state for the deleted project
                    st.session_state.project_id = None
                    st.session_state.project_name = None
                    st.session_state.project_location = None
                    st.session_state.analysis_summary_text = None
                    st.session_state.safety_violations = None
                    st.session_state.processed_media_path = None
                    st.session_state.processed_media_type = None
                    st.session_state.original_uploaded_filename = None
                    st.session_state.llm_generated_report = None
                    st.session_state.pdf_preview_path = None
                    st.session_state.pop('confirm_delete_project', None) # Clear confirmation state
                    st.rerun() # Rerun to update project list and clear details
                else:
                    st.error(f"Failed to delete project '{st.session_state.project_name}'.")
                    st.session_state.pop('confirm_delete_project', None) # Clear confirmation state on failure

else:
    st.info("No project is currently selected or created. Please create a new one or select from the list.")

# --- Sidebar Display ---
st.sidebar.header("Current Project")
if st.session_state.project_name and st.session_state.project_location:
    st.sidebar.success(f"Project: **{st.session_state.project_name}**")
    st.sidebar.info(f"Location: {st.session_state.project_location}")
else:
    st.sidebar.warning("No project created or selected.")

st.sidebar.divider()
st.sidebar.info("Navigate using the pages menu above.")

# Clean up confirmation state if not acted upon on rerun
if 'confirm_delete_project' in st.session_state and not delete_confirm and not st.session_state.get('confirm_delete_now_btn'):
     st.session_state.pop('confirm_delete_project', None)