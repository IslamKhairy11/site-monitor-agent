# -----------------------------------------------------------------------------
# utils/reporting.py - LLM Summary and PDF Generation
# -----------------------------------------------------------------------------

import time
import io
import streamlit as st
# import google.generativeai as genai # Not needed directly here anymore
# import openai # Not needed directly here anymore
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader # For embedding images
import os
import json # For serializing violations if needed, though not strictly required for PDF text

# Import the LLM manager
from utils.llm_manager import get_llm_model, generate_text

# -----------------------------------------------------------------------------
# 1. Generate Report Text using LLM (via llm_manager)
# -----------------------------------------------------------------------------

def generate_llm_report(project_name, project_location, analysis_summary_text, violations, llm_type, api_key):
    """
    Generates report content (findings and recommendations) using the selected LLM model.

    Args:
        project_name (str): Name of the project.
        project_location (str): Location of the project.
        analysis_summary_text (str): Text summary from image/video analysis.
        violations (list): List of structured violation dictionaries.
        llm_type (str): The selected LLM type ("Gemini" or "OpenAI").
        api_key (str): The API key for the selected LLM.

    Returns:
        dict: A dictionary containing 'findings' and 'recommendations' strings,
              or None if LLM fails. Includes an 'error' key if failure occurs.
    """
    print(f"Generating LLM Summary Report using {llm_type}...")

    model_info, error = get_llm_model(llm_type, api_key)
    if model_info is None:
        return {
            "error": error,
            "findings": "Error: LLM not configured or failed.",
            "recommendations": "Unable to generate recommendations without the LLM."
        }

    # Format violations nicely for the LLM prompt
    violations_text = "No specific safety violations detected."
    if violations:
        violations_text = "Potential Safety Violations Detected (Sample):\n"
        # Only send a sample or summary of violations to the LLM if the list is very long
        sample_violations = violations[:10] + (["..."] if len(violations) > 10 else [])
        for i, vio in enumerate(sample_violations):
             if isinstance(vio, str): # Handle the "..." case
                 violations_text += f"- {vio}\n"
             else:
                # Include key details like class and confidence, maybe frame/timestamp
                vio_str = f"- {vio['class']} (Confidence: {vio['confidence']:.2f})"
                if 'frame' in vio and 'timestamp' in vio:
                    vio_str += f" at Frame {vio['frame']} ({vio['timestamp']})"
                violations_text += vio_str + "\n"
        if len(violations) > 10:
             violations_text += f"({len(violations) - 10} more violations found)\n"


    prompt = f"""
    You are a construction site AI monitor. Generate ONLY the "Key Findings" and "Recommendations" sections of a report based on the following analysis data. Do NOT include project details, date, or the raw analysis summary in your response. Start directly with "Key Findings:" followed by findings, then "Recommendations:" followed by recommendations.

    Project Name: '{project_name}'
    Project Location: '{project_location}'
    Date of Analysis: {time.strftime("%Y-%m-%d %H:%M:%S")}

    Automated analysis summary:
    ```
    {analysis_summary_text}
    ```

    Specific details about potential safety violations detected:
    ```
    {violations_text}
    ```

    Based on this data, provide:
    1.  Key Findings: Summarize the most significant observations, especially safety issues.
    2.  Recommendations: Provide actionable suggestions for site improvement based on the findings.

    Format your response strictly as:
    Key Findings:
    - Finding 1
    - Finding 2
    ...
    Recommendations:
    - Recommendation 1
    - Recommendation 2
    ...
    """

    try:
        llm_raw_response = generate_text(model_info, prompt, llm_type)
        print(f"LLM Raw Response:\n{llm_raw_response}")

        # Parse the raw response to extract findings and recommendations
        findings = ""
        recommendations = ""
        current_section = None

        # Split by lines and process
        lines = llm_raw_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("key findings:"):
                current_section = "findings"
                findings = line[len("key findings:"):].strip() + "\n" # Start with rest of the line
            elif line.lower().startswith("recommendations:"):
                current_section = "recommendations"
                recommendations = line[len("recommendations:"):].strip() + "\n" # Start with rest of the line
            elif current_section == "findings":
                findings += line + "\n"
            elif current_section == "recommendations":
                recommendations += line + "\n"

        # Clean up leading/trailing whitespace
        findings = findings.strip()
        recommendations = recommendations.strip()

        return {
            "findings": findings,
            "recommendations": recommendations
        }

    except Exception as e:
        print(f"Error during LLM report generation: {e}")
        return {
            "error": f"An error occurred during LLM text generation: {e}",
            "findings": "Error generating findings.",
            "recommendations": "Error generating recommendations."
        }

# -----------------------------------------------------------------------------
# 2. Create PDF Report from Generated Text and Analysis Data
# -----------------------------------------------------------------------------

def create_pdf_report(project_name: str, project_location: str, analysis_summary_text: str, violations: list, llm_generated_report: dict, annotated_image_path: str = None, include_violations: bool = True, include_image: bool = True, body_font_size: int = 10, heading1_font_size: int = 18, heading2_font_size: int = 14, image_width_inch: float = 6):
    """
    Converts report text, analysis summary, violations, and an optional image
    into a styled PDF, with customization options.

    Args:
        project_name (str): Name of the project.
        project_location (str): Location of the project.
        analysis_summary_text (str): Raw text summary from analysis.
        violations (list): Structured violation data.
        llm_generated_report (dict): Dictionary with 'findings' and 'recommendations' from LLM.
        annotated_image_path (str, optional): Path to the annotated image file (for image analysis). Defaults to None.
        include_violations (bool): Whether to include the detailed violations list. Defaults to True.
        include_image (bool): Whether to include the annotated image (if image analysis). Defaults to True.
        body_font_size (int): Font size for normal text.
        heading1_font_size (int): Font size for main headings.
        heading2_font_size (int): Font size for subheadings.
        image_width_inch (float): Desired width for the annotated image in inches.


    Returns:
        bytes: The PDF content as bytes, or None if an error occurs.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    # Create/Modify styles based on customization parameters
    # Use base styles and adjust font size and leading (line spacing)
    normal_style = ParagraphStyle(name='NormalCustom',
                                  parent=styles['Normal'],
                                  fontSize=body_font_size,
                                  leading=body_font_size * 1.2) # Simple leading adjustment

    heading1_style = ParagraphStyle(name='Heading1Custom',
                                    parent=styles['Heading1'],
                                    fontSize=heading1_font_size,
                                    leading=heading1_font_size * 1.1,
                                    spaceAfter=heading1_font_size * 0.5) # Add space after heading

    heading2_style = ParagraphStyle(name='Heading2Custom',
                                    parent=styles['Heading2'],
                                    fontSize=heading2_font_size,
                                    leading=heading2_font_size * 1.1,
                                    spaceAfter=heading2_font_size * 0.4) # Add space after subheading

    # Use the 'Code' style for the raw analysis summary block
    # We can adjust its size too if needed, but often default Code style is fixed-width and fine.
    # Let's create a custom code style inheriting from 'Code' if we want to adjust size.
    code_style = ParagraphStyle(name='CodeCustom',
                                parent=styles['Code'],
                                fontSize=body_font_size * 0.9, # Slightly smaller than body
                                leading=body_font_size * 1.1)


    story = []

    # Title
    story.append(Paragraph("Construction Site Analysis Report", heading1_style))
    story.append(Spacer(1, 0.2*inch))

    # Project Details
    story.append(Paragraph(f"<b>Project:</b> {project_name}", normal_style))
    story.append(Paragraph(f"<b>Location:</b> {project_location}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3*inch))

    # Original Analysis Summary
    story.append(Paragraph("Raw Analysis Summary", heading2_style))
    story.append(Spacer(1, 0.1*inch))
    # Use the custom code style for the raw summary text block
    story.append(Preformatted(analysis_summary_text.strip(), code_style))
    story.append(Spacer(1, 0.3*inch))

    # Potential Safety Violations (Detailed List)
    if include_violations and violations:
        story.append(Paragraph("Potential Safety Violations Detected", heading2_style))
        story.append(Spacer(1, 0.1*inch))
        for i, vio in enumerate(violations):
            # Format violation details
            vio_str = f"- {vio['class']} (Confidence: {vio['confidence']:.2f})"
            if 'frame' in vio and 'timestamp' in vio:
                vio_str += f" at Frame {vio['frame']} ({vio['timestamp']})"
            # Optionally add bounding box info if desired
            # if 'bbox' in vio:
            #      bbox = vio['bbox']
            #      # Assuming bbox is {'x': ..., 'y': ..., 'width': ..., 'height': ...} or [x1, y1, x2, y2]
            #      if isinstance(bbox, dict):
            #          vio_str += f" Bbox: ({bbox['x']:.0f},{bbox['y']:.0f},{bbox['width']:.0f},{bbox['height']:.0f})"
            #      elif isinstance(bbox, list) and len(bbox) == 4:
            #           vio_str += f" Bbox: ({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"


            story.append(Paragraph(vio_str, normal_style))
        story.append(Spacer(1, 0.3*inch))
    elif include_violations: # If user wants violations but none were detected
         story.append(Paragraph("Potential Safety Violations Detected", heading2_style))
         story.append(Spacer(1, 0.1*inch))
         story.append(Paragraph("No specific safety violations were detected in the analysis.", normal_style))
         story.append(Spacer(1, 0.3*inch))


    # Annotated Image (for image analysis)
    if include_image and annotated_image_path and os.path.exists(annotated_image_path):
        try:
            img = Image(annotated_image_path, width=image_width_inch * inch) # Set width based on input
            # Maintain aspect ratio based on desired width
            img_width, img_height = img.drawWidth, img.drawHeight # Get actual size after setting width
            aspect = img_height / float(img_width)
            # img.drawWidth is already set by the constructor, just need to scale height
            img.drawHeight = img.drawWidth * aspect

            # Add image only if it can be loaded and scaled reasonably
            if img.drawWidth > 0 and img.drawHeight > 0:
                story.append(Paragraph("Analyzed Media Snapshot", heading2_style))
                story.append(Spacer(1, 0.1*inch))
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
            else:
                 print(f"Warning: Image {annotated_image_path} could not be scaled properly.")
                 story.append(Paragraph(f"Could not include analyzed image snapshot.", normal_style))
                 story.append(Spacer(1, 0.3*inch))

        except Exception as e:
            print(f"Error adding image {annotated_image_path} to PDF: {e}")
            story.append(Paragraph(f"Error including analyzed image snapshot: {e}", normal_style))
            story.append(Spacer(1, 0.3*inch))
    elif include_image: # If user wants image but it's not applicable (video) or path is bad
         story.append(Paragraph("Analyzed Media Snapshot", heading2_style))
         story.append(Spacer(1, 0.1*inch))
         media_note = "Analysis was performed on video. A single representative frame is not included in this PDF." if annotated_image_path is None else "Could not include analyzed image snapshot."
         story.append(Paragraph(media_note, normal_style))
         story.append(Spacer(1, 0.3*inch))


    # LLM Generated Report Sections (Key Findings, Recommendations)
    findings_text = llm_generated_report.get('findings', 'No findings generated.')
    recommendations_text = llm_generated_report.get('recommendations', 'No recommendations generated.')
    llm_error = llm_generated_report.get('error')

    if llm_error:
         story.append(Paragraph(f"LLM Report Generation Error: {llm_error}", normal_style))
         story.append(Spacer(1, 0.3*inch))

    # Add Key Findings
    story.append(Paragraph("Key Findings", heading2_style))
    story.append(Spacer(1, 0.1*inch))
    # Split findings text by lines to create separate paragraphs
    for line in findings_text.strip().split('\n'):
         if line.strip(): # Only add non-empty lines
             story.append(Paragraph(line.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))

    # Add Recommendations
    story.append(Paragraph("Recommendations", heading2_style))
    story.append(Spacer(1, 0.1*inch))
    # Split recommendations text by lines
    for line in recommendations_text.strip().split('\n'):
         if line.strip(): # Only add non-empty lines
            story.append(Paragraph(line.strip(), normal_style))
    story.append(Spacer(1, 0.3*inch))


    # Build the PDF
    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        print(f"Error building PDF: {e}")
        buffer.close()
        return None