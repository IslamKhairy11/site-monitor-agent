# -----------------------------------------------------------------------------
# pages/01_Site_Analysis.py - Site Analysis Page
# Handles file uploads, analysis, results display, report generation & preview.
# -----------------------------------------------------------------------------

import streamlit as st
import os
import time
import json
import base64 # For PDF embedding
import tempfile # For temporary PDF file
import shutil # For cleaning up old processed media

# Import functions from utils
from utils.analysis import analyze_image_with_inference_pkg, analyze_video_with_inference_pkg
from utils.reporting import generate_llm_report, create_pdf_report
from utils.database import DATA_DIR # Import DATA_DIR constant
from utils.llm_manager import get_llm_model # To check LLM availability


# --- Page Content ---
st.title("📊 AI Construction Site Analysis")

# --- Check Project Setup ---
if not st.session_state.get('project_id'):
    st.warning("⚠️ Please create or select a project on the main page first!")
    st.stop()

st.info(f"Analyzing for Project: **{st.session_state.project_name}** (ID: {st.session_state.project_id})")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload an Image (jpg, png) or Video (mp4, mov, avi)",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    key="file_uploader_analysis" # Unique key
)

# --- Analysis Execution ---
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    original_filename = uploaded_file.name # Get original filename
    file_type = uploaded_file.type

    # If a new file is uploaded or project changed, clear previous analysis/reporting results
    # Check if project_id matches the one associated with the last upload
    if st.session_state.get('uploaded_file_info', {}).get('project_id') != st.session_state.project_id or \
       st.session_state.get('uploaded_file_info', {}).get('name') != original_filename or \
       st.session_state.get('uploaded_file_info', {}).get('type') != file_type:

        print(f"New file uploaded or project changed. Clearing previous state.")
        st.session_state.uploaded_file_info = {'name': original_filename, 'type': file_type, 'project_id': st.session_state.project_id}
        st.session_state.analysis_summary_text = None
        st.session_state.safety_violations = None
        st.session_state.processed_media_path = None # This will be set *after* analysis
        st.session_state.processed_media_type = None # This will be set *after* analysis
        st.session_state.llm_generated_report = None # Clear old LLM report
        st.session_state.pdf_preview_path = None # Clear old PDF preview
        st.rerun() # Rerun to clear displayed results

    st.write(f"Uploaded: **{original_filename}** ({file_type})")

    analyze_button = None
    if file_type.startswith("image/"):
        st.image(file_bytes, caption="Original Image", use_column_width=True)
        analyze_button = st.button("Analyze Image", key="analyze_image_btn")

    elif file_type.startswith("video/"):
        st.video(file_bytes) # Display the original video
        analyze_button = st.button("Analyze Video", key="analyze_video_btn")

    else:
        st.error("Unsupported file type.")
        st.stop()

    # --- Trigger Analysis ---
    if analyze_button:
        # Clean up previously processed media file for this project *if it was from a previous analysis run*
        # We only clean up the *last* processed file associated with this project in session state
        if st.session_state.processed_media_path and os.path.exists(st.session_state.processed_media_path):
            try:
                # Ensure the path is within the current project's processed_media directory before deleting
                expected_dir = os.path.join(DATA_DIR, str(st.session_state.project_id), 'processed_media')
                if os.path.abspath(st.session_state.processed_media_path).startswith(os.path.abspath(expected_dir)):
                     os.remove(st.session_state.processed_media_path)
                     print(f"Cleaned up old processed media: {st.session_state.processed_media_path}")
                else:
                     print(f"Warning: Processed media path {st.session_state.processed_media_path} is outside expected directory {expected_dir}. Not auto-deleting.")
            except Exception as e:
                print(f"Error cleaning up old processed media {st.session_state.processed_media_path}: {e}")

        # Clear analysis and reporting state for the new analysis run
        st.session_state.analysis_summary_text = None
        st.session_state.safety_violations = None
        st.session_state.processed_media_path = None
        st.session_state.processed_media_type = None
        st.session_state.llm_generated_report = None
        st.session_state.pdf_preview_path = None

        with st.spinner(f"Running analysis on {original_filename}... This might take a while for videos."):
            if file_type.startswith("image/"):
                st.session_state.processed_media_type = 'image_annotated'
                processed_path, analysis_details = analyze_image_with_inference_pkg(
                    file_bytes,
                    st.session_state.project_id,
                    original_filename
                )
            elif file_type.startswith("video/"):
                st.session_state.processed_media_type = 'video_annotated'
                processed_path, analysis_details = analyze_video_with_inference_pkg(
                    file_bytes,
                    st.session_state.project_id,
                    original_filename
                )
            else:
                 processed_path, analysis_details = None, {"summary_text": "Unsupported file type.", "violations": []}


            if processed_path and analysis_details:
                st.session_state.processed_media_path = processed_path
                st.session_state.analysis_summary_text = analysis_details.get("summary_text", "Analysis complete.")
                st.session_state.safety_violations = analysis_details.get("violations", [])
                st.success("Analysis Complete!")
            else:
                st.error(f"Analysis failed: {analysis_details.get('summary_text', 'Unknown error.')}")
                st.session_state.processed_media_path = None
                st.session_state.processed_media_type = None
                st.session_state.analysis_summary_text = analysis_details.get("summary_text", "Analysis failed.")
                st.session_state.safety_violations = []

        st.rerun() # Rerun to display results sections


# --- Display Analysis Results and Generate Report ---
st.divider()
st.subheader("Analysis Results & Reporting")

# Retrieve results from session state
analysis_summary_text = st.session_state.get('analysis_summary_text', None)
safety_violations = st.session_state.get('safety_violations', [])
processed_media_path = st.session_state.get('processed_media_path', None)
current_media_type = st.session_state.get('processed_media_type', None)
# original_filename is already stored in session_state.uploaded_file_info['name'] if needed


if analysis_summary_text: # Display results if analysis has been performed
    st.write(f"**Last Analysis Performed:** {current_media_type.replace('_', ' ').title() if current_media_type else 'Unknown'}")

    # --- Display Processed Output ---
    st.subheader("Processed Media")
    if processed_media_path and os.path.exists(processed_media_path):
        try:
            if current_media_type == 'image_annotated':
                st.image(processed_media_path, caption="Analyzed Image", use_column_width=True)
                with open(processed_media_path, "rb") as f:
                     img_bytes = f.read()
                st.download_button(
                    label="Download Analyzed Image",
                    data=img_bytes,
                    file_name=os.path.basename(processed_media_path),
                    mime="image/png"
                )
            elif current_media_type == 'video_annotated':
                st.info(f"Annotated video saved: {os.path.basename(processed_media_path)}")
                # Display the video by reading bytes
                try:
                     with open(processed_media_path, "rb") as video_file:
                         video_file_bytes = video_file.read()
                     st.video(video_file_bytes, format = "video/mp4")
                     # Offer for download
                     st.download_button(
                         label="Download Analyzed Video",
                         data=video_file_bytes,
                         file_name=os.path.basename(processed_media_path),
                         mime="video/mp4"
                     )
                except FileNotFoundError:
                     st.error("Error: Annotated video file not found for preview.")
                except Exception as e:
                     st.error(f"Could not display or offer video for download: {e}")
            else:
                 st.warning("Processed media path is set, but the file type is unknown or unsupported for display.")

        except Exception as e:
            st.error(f"Error displaying processed media: {e}")
    else:
         st.info("Processed media file not available.")


    # --- Display Analysis Summary ---
    st.divider()
    st.subheader("Analysis Summary Text")
    st.text_area("Raw Analysis Summary:", value=analysis_summary_text, height=200, disabled=True, key="analysis_summary_area")

    # Display detected violations separately for clarity
    if safety_violations:
        st.subheader("Detected Potential Safety Violations")
        violation_display_limit = 10 # Limit display in Streamlit for long lists
        for i, vio in enumerate(safety_violations[:violation_display_limit]):
             vio_str = f"- **{vio['class']}** (Confidence: {vio['confidence']:.2f})"
             if 'frame' in vio and 'timestamp' in vio:
                 vio_str += f" at Frame {vio['frame']} (Time: {vio['timestamp']})"
             st.markdown(vio_str)
        if len(safety_violations) > violation_display_limit:
             st.markdown(f"... {len(safety_violations) - violation_display_limit} more violations detected.")
    else:
        st.info("No specific safety violations detected in the analysis.")


    # --- Reporting Section ---
    st.divider()
    st.subheader("Generate Report")

    llm_available = st.session_state.get('llm_api_key') is not None and st.session_state.get('llm_api_key') != ""
    if not llm_available:
         st.warning("Please configure your LLM API key on the main page to generate LLM reports.")

    # --- LLM Report Generation ---
    if llm_available:
        if st.button("Generate LLM Report Summary", key="generate_llm_report_btn"):
            with st.spinner(f"Generating report summary using {st.session_state.selected_llm_type}..."):
                llm_report_data = generate_llm_report(
                    st.session_state.project_name,
                    st.session_state.project_location,
                    analysis_summary_text,
                    safety_violations, # Pass structured violations
                    st.session_state.selected_llm_type,
                    st.session_state.llm_api_key
                )
                st.session_state.llm_generated_report = llm_report_data
                st.session_state.pdf_preview_path = None # Clear old preview when report text changes
                st.rerun()

    # --- Display LLM Report Summary ---
    llm_generated_report = st.session_state.get('llm_generated_report', None)
    if llm_generated_report:
        st.subheader("Generated Report Summary (LLM)")
        if llm_generated_report.get('error'):
             st.error(f"LLM Error: {llm_generated_report['error']}")

        st.markdown("**Key Findings:**")
        if llm_generated_report.get('findings'):
            # Display findings line by line
            for line in llm_generated_report['findings'].split('\n'):
                 if line.strip():
                     st.markdown(line.strip())
        else:
             st.info("No findings generated by the LLM.")

        st.markdown("**Recommendations:**")
        if llm_generated_report.get('recommendations'):
            # Display recommendations line by line
            for line in llm_generated_report['recommendations'].split('\n'):
                 if line.strip():
                     st.markdown(line.strip())
        else:
             st.info("No recommendations generated by the LLM.")


        # --- PDF Generation and Preview ---
        st.divider()
        st.subheader("PDF Report Generation")

        # PDF Customization Options
        st.markdown("Configure your PDF report content and appearance:")
        col_report_opts1, col_report_opts2, col_report_opts3 = st.columns(3)
        with col_report_opts1:
            include_violations = st.checkbox("Include Detailed Safety Violations List", value=st.session_state.get('report_include_violations', True), key="report_include_violations")
            # Only show image option for image analysis, or if a placeholder is desired for video
            include_image_default = st.session_state.get('report_include_image', (current_media_type == 'image_annotated')) # Default to checked if image, or last saved state
            include_image = st.checkbox("Include Analyzed Media Snapshot", value=include_image_default, key="report_include_image")

        with col_report_opts2:
            # Text size customization
            body_font_size = st.number_input(
                 "Body Text Size (pt)",
                 min_value=8, max_value=14, value=st.session_state.get('pdf_body_font_size', 10), step=1,
                 key="pdf_body_font_size_input",
                 help="Font size for paragraphs and list items."
            )
            heading2_font_size = st.number_input(
                 "Subheading Size (pt)",
                 min_value=12, max_value=20, value=st.session_state.get('pdf_heading2_font_size', 14), step=1,
                 key="pdf_heading2_font_size_input",
                 help="Font size for sections like 'Analysis Summary', 'Key Findings'."
            )
        with col_report_opts3:
            heading1_font_size = st.number_input(
                 "Title Size (pt)",
                 min_value=16, max_value=24, value=st.session_state.get('pdf_heading1_font_size', 18), step=1,
                 key="pdf_heading1_font_size_input",
                 help="Font size for the main report title."
            )
            # Retrieve image width with type check
            image_width_inch_value = st.session_state.get('pdf_image_width_inch', 6.0)
            if isinstance(image_width_inch_value, list):
                 print(f"Warning: session_state.pdf_image_width_inch was list {image_width_inch_value}. Resetting to default 6.0.")
                 image_width_inch_value = 6.0 # Reset to a valid float default if it's a list

            image_width_inch = st.slider(
                 "Image Width (inches)",
                 min_value=3.0, max_value=7.5, value=float(image_width_inch_value), step=0.5, # Ensure value is float
                 key="pdf_image_width_inch_input",
                 help="Width of the annotated image in the PDF (aspect ratio is maintained)."
            )

        # Update session state with customization values
        st.session_state.pdf_body_font_size = body_font_size
        st.session_state.pdf_heading1_font_size = heading1_font_size
        st.session_state.pdf_heading2_font_size = heading2_font_size
        st.session_state.pdf_image_width_inch = image_width_inch


        # Generate PDF button
        if st.button("Generate PDF Report", key="generate_pdf_btn"):
             st.session_state.pdf_preview_path = None # Clear old preview before generating new one
             with st.spinner("Creating PDF..."):
                # Determine image path to pass to PDF function
                img_path_for_pdf = processed_media_path if current_media_type == 'image_annotated' else None

                pdf_bytes = create_pdf_report(
                    st.session_state.project_name,
                    st.session_state.project_location,
                    analysis_summary_text,
                    safety_violations,
                    llm_generated_report, # Pass the dictionary result
                    annotated_image_path=img_path_for_pdf, # Pass image path regardless of include_image, function handles if file exists
                    include_violations=include_violations,
                    include_image=include_image, # Pass the user's preference
                    body_font_size=body_font_size,
                    heading1_font_size=heading1_font_size,
                    heading2_font_size=heading2_font_size,
                    image_width_inch=image_width_inch
                )

                if pdf_bytes:
                    # Save PDF temporarily for preview and download
                    report_dir = os.path.join(DATA_DIR, str(st.session_state.project_id), 'reports')
                    os.makedirs(report_dir, exist_ok=True) # Ensure report directory exists
                    # Use a timestamp in the filename to avoid conflicts and allow multiple reports
                    temp_pdf_filename = f"{st.session_state.project_name.replace(' ', '_')}_Analysis_Report_{int(time.time())}.pdf"
                    temp_pdf_path = os.path.join(report_dir, temp_pdf_filename)

                    try:
                        with open(temp_pdf_path, "wb") as f:
                            f.write(pdf_bytes)
                        st.session_state.pdf_preview_path = temp_pdf_path # Store path for display/download
                        st.success("PDF report generated.")
                    except Exception as e:
                        st.error(f"Failed to save PDF for preview/download: {e}")
                        st.session_state.pdf_preview_path = None # Clear path on error

                else:
                    st.error("Failed to generate PDF report.")

             st.rerun() # Rerun to show preview/download

        # Display PDF Preview and Download
        pdf_preview_path = st.session_state.get('pdf_preview_path', None)
        if pdf_preview_path and os.path.exists(pdf_preview_path):
             st.subheader("PDF Preview")
             try:
                 with open(pdf_preview_path, "rb") as f:
                     pdf_bytes_for_display = f.read()

                 # Embed PDF using Base64
                 base64_pdf = base64.b64encode(pdf_bytes_for_display).decode('utf-8')
                 # Use the custom CSS class for the iframe container
                 pdf_display = f'<div class="pdf-iframe-container"><iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf"></iframe></div>'
                 st.markdown(pdf_display, unsafe_allow_html=True)

                 # Download button
                 st.download_button(
                     label="Download PDF Report",
                     data=pdf_bytes_for_display,
                     file_name=os.path.basename(pdf_preview_path),
                     mime="application/pdf"
                 )
             except FileNotFoundError:
                 st.error("PDF file not found for preview.")
             except Exception as e:
                 st.error(f"Error displaying PDF preview: {e}")

    elif analysis_summary_text and not llm_generated_report and llm_available:
        st.info("Click 'Generate LLM Report Summary' to create the report text.")
    elif not analysis_summary_text:
         st.info("Upload an image or video file above and click 'Analyze' to begin.")