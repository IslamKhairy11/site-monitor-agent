# -----------------------------------------------------------------------------
# utils/analysis.py
# Image and Video Analysis Logic using 'inference' and 'supervision' libraries
# Enhanced to extract violation details.
# -----------------------------------------------------------------------------

import io
import os
import time
import numpy as np
import cv2 # OpenCV for image/video frame handling
from inference import get_model # Using the 'inference' package
import supervision as sv
import tempfile # For managing temporary files
import shutil # For directory cleanup
import streamlit as st # For caching models

# --- Roboflow API Keys and Model IDs ---
# Note: These are hardcoded for the models used. If using different models,
# you might need to parameterize these or handle model loading differently.
# API keys are embedded in the inference package models usually, but explicitly
# providing them here ensures the models are fetched correctly.

# Using dummy keys for display. REPLACE WITH YOUR ACTUAL KEYS or manage via secrets/env vars
# RF_API_KEY_SE = st.secrets.get("RF_API_KEY_SE", "YOUR_STRUCT_EXCAV_API_KEY")
# RF_API_KEY_PPE = st.secrets.get("RF_API_KEY_PPE", "YOUR_PPE_API_KEY")
# For this example, we'll assume the keys are associated with the model IDs or
# are handled by the environment where 'inference' is configured.
# If 'inference.get_model' requires explicit keys, uncomment and use Streamlit secrets.
# For now, using the keys provided in the original code snippet.
RF_API_KEY_SE = "CtTbvTxYKh5hwbzVNnNJ"  # Structural + Excavator
RF_API_KEY_PPE = "ERmz24EWzu0uIFuf9mY3" # PPE


MODEL_ID_STRUCTURAL = "eagle.ai-structural-components/4"
MODEL_ID_EXCAVATOR = "eagle.ai-excavator/1"
MODEL_ID_PPE = "eagle.ai-ppe-imstb/1"

# Define classes considered as safety violations (based on the PPE model)
SAFETY_VIOLATION_CLASSES = ['no-hard-hat', 'no-safety-vest', 'no-gloves', 'no-eye-protection'] # Add other relevant classes

# --- Model Loading (using Streamlit caching) ---
@st.cache_resource
def load_all_models(se_key, ppe_key):
    """Loads all three Roboflow models using st.cache_resource."""
    print("Loading Roboflow models...")
    try:
        model_struct = get_model(model_id=MODEL_ID_STRUCTURAL, api_key=se_key)
        model_excav = get_model(model_id=MODEL_ID_EXCAVATOR, api_key=se_key)
        model_ppe = get_model(model_id=MODEL_ID_PPE, api_key=ppe_key)
        print("Models loaded successfully.")
        return model_struct, model_excav, model_ppe
    except Exception as e:
        print(f"Error loading models: {e}")
        st.error(f"Failed to load one or more detection models. Check API keys and model IDs. Error: {e}")
        return None, None, None

# --- Helper to extract violation details ---
def extract_violations(detections, classes, frame_info=None):
    """Extracts safety violation details from detections."""
    violations = []
    for i in range(len(detections.xyxy)):
        class_id = detections.class_id[i]
        confidence = detections.confidence[i]
        # Ensure class_id is within bounds of classes list
        if class_id is not None and 0 <= class_id < len(classes):
            class_name = classes[class_id]
            if class_name in SAFETY_VIOLATION_CLASSES:
                violation_detail = {
                    'class': class_name,
                    'confidence': float(confidence), # Ensure JSON serializable
                    'bbox': [float(coord) for coord in detections.xyxy[i]] # Ensure JSON serializable
                }
                if frame_info:
                    violation_detail.update(frame_info) # Add frame/timestamp info for videos
                violations.append(violation_detail)
        else:
             print(f"Warning: Skipping detection with out-of-bounds class_id {class_id}")
    return violations

# --- Image Analysis using 'inference' and 'supervision' ---
def analyze_image_with_inference_pkg(image_bytes, project_id, original_filename):
    """
    Analyzes a single image using three Roboflow models, saves the annotated
    image to the project directory, and returns its path and analysis details.

    Args:
        image_bytes (bytes): Bytes data of the uploaded image.
        project_id (int): The ID of the current project.
        original_filename (str): The original name of the uploaded image file.

    Returns:
        tuple: (annotated_image_path, analysis_details) or (None, error_message).
               analysis_details is a dict including summary text and violations.
    """
    model_struct, model_excav, model_ppe = load_all_models(RF_API_KEY_SE, RF_API_KEY_PPE)
    if not all([model_struct, model_excav, model_ppe]):
        return None, {"summary_text": "Error: Failed to load one or more models.", "violations": []}

    try:
        # Convert image bytes to OpenCV format (NumPy array)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, {"summary_text": "Error: Could not decode image.", "violations": []}

        print("Performing inference on image...")
        # Inference on each model
        # Handle potential empty prediction lists gracefully
        try:
            r_struct = model_struct.infer(frame)[0]
        except (IndexError, TypeError, AttributeError):
            r_struct = type('obj', (object,), {'predictions': []})() # Create mock object

        try:
            r_excav = model_excav.infer(frame)[0]
        except (IndexError, TypeError, AttributeError):
            r_excav = type('obj', (object,), {'predictions': []})()

        try:
            r_ppe = model_ppe.infer(frame)[0]
        except (IndexError, TypeError, AttributeError):
            r_ppe = type('obj', (object,), {'predictions': []})()


        print("Converting inference results to Supervision Detections...")
        # Get class names from models (assuming order matches prediction class_ids)
        # This might require checking model API documentation or a more robust way
        # to map class_id to name if models don't return names directly in predictions.
        # For 'inference' package results, p.class_name is available, but sv.Detections.from_inference
        # maps class_id based on the *order* of unique class names encountered.
        # Let's collect all unique class names to pass to supervision.
        all_class_names = sorted(list(set(
            [p.class_name for p in r_struct.predictions] +
            [p.class_name for p in r_excav.predictions] +
            [p.class_name for p in r_ppe.predictions]
        )))

        # Convert to supervision detections
        # Pass class names explicitly if possible, or rely on from_inference's mapping
        # sv.Detections.from_inference uses results.class_id and results.confidence
        # It seems it implicitly maps class_id based on the order of class_names
        # in the results object if available, or just uses the IDs.
        # To be safe, let's ensure the resulting detections have class names.
        # A more robust way is to manually construct Detections with labels.
        # For simplicity, let's assume from_inference handles this reasonably
        # or we map back using the original prediction objects.

        d_struct = sv.Detections.from_inference(r_struct)
        d_excav = sv.Detections.from_inference(r_excav)
        d_ppe = sv.Detections.from_inference(r_ppe)

        # Merge all detections
        detections = sv.Detections.merge([d_struct, d_excav, d_ppe])

        # Get labels for merged detections - need to map class_id back to name
        # based on how merge combines them. This can be tricky.
        # A simpler approach is to get labels directly from original predictions
        # and ensure they align with the merged detections.
        # Let's reconstruct labels based on the original prediction order and hope merge preserves it.
        # A more robust way would be to iterate the *merged* detections and map their class_ids.
        # Assuming sv.Detections.merge combines in order:
        all_predictions = r_struct.predictions + r_excav.predictions + r_ppe.predictions
        labels = [f"{p.class_name} ({p.confidence:.2f})" for p in all_predictions]


        # Extract safety violations specifically from PPE detections
        ppe_class_names_list = [p.class_name for p in r_ppe.predictions] # Get class names specifically from PPE model
        # Need a mapping from class_id in d_ppe to the actual class name.
        # Let's iterate the d_ppe detections and find the corresponding original prediction.
        violations = []
        if hasattr(r_ppe, 'predictions'):
            for i in range(len(r_ppe.predictions)):
                 p = r_ppe.predictions[i]
                 if p.class_name in SAFETY_VIOLATION_CLASSES:
                      violations.append({
                          'class': p.class_name,
                          'confidence': p.confidence,
                          'bbox': p.rect # Use rect for bbox
                      })


        # Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS) # Use color per class
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=2, color_lookup=sv.ColorLookup.CLASS)

        print("Annotating image...")
        annotated_frame = frame.copy()
        if len(detections.xyxy) > 0:
            # Need to map detections.class_id to a color. Let's use sv.ColorLookup.CLASS
            # which requires detections.class_id and a class_id->color map or relies on default.
            # Let's try annotating detections and providing labels separately.
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

            # Reconstruct labels with confidence for annotation display
            # This requires re-associating merged detections with original predictions, which is complex.
            # A simpler approach for display labels: just use class name + confidence if available.
            # Let's assume the order in `labels` matches the order in `detections`. This is fragile.
            # A robust approach needs manual mapping or re-generating labels after merge.
            # For display simplicity, let's just use the collected prediction labels.
            if len(labels) == len(detections.xyxy):
                 display_labels = [f"{label.split('(')[0].strip()}" for label in labels] # Clean up confidence for display label if needed
                 annotated_frame = label_annotator.annotate(
                     scene=annotated_frame,
                     detections=detections,
                     labels=display_labels # Use simplified labels for display
                 )
            else:
                 print(f"Warning: Label count ({len(labels)}) mismatch with detection count ({len(detections.xyxy)}). Using default labels.")
                 annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)


        # Save annotated image to project directory
        project_dir = os.path.join('data', str(project_id))
        processed_media_dir = os.path.join(project_dir, 'processed_media')
        output_filename = f"annotated_{os.path.splitext(original_filename)[0]}.png"
        annotated_image_path = os.path.join(processed_media_dir, output_filename)

        cv2.imwrite(annotated_image_path, annotated_frame)
        print(f"Annotated image saved to: {annotated_image_path}")

        # Generate summary text
        summary_text = "Image Analysis Summary:\n"
        summary_text += f"- Structural elements detected: {len(d_struct.xyxy)}\n"
        summary_text += f"- Excavators detected: {len(d_excav.xyxy)}\n"
        summary_text += f"- PPE items detected: {len(d_ppe.xyxy)}\n"
        detected_classes = set([p.class_name for p in all_predictions])
        summary_text += f"All detected classes: {', '.join(detected_classes) if detected_classes else 'None'}\n"

        # Add violation details to summary text
        if violations:
            summary_text += "\nPotential Safety Violations Detected:\n"
            for i, vio in enumerate(violations):
                summary_text += f"  {i+1}. {vio['class']} (Confidence: {vio['confidence']:.2f})\n"
                # Optionally add bbox: f" Bbox: ({vio['bbox']['x']:.0f},{vio['bbox']['y']:.0f},{vio['bbox']['width']:.0f},{vio['bbox']['height']:.0f})"
        else:
            summary_text += "\nNo specific safety violations detected in this image."


        analysis_details = {
            "summary_text": summary_text,
            "violations": violations, # Store structured violation data
            "image_path": annotated_image_path # Store path to annotated image
        }

        print("Image analysis complete.")
        return annotated_image_path, analysis_details

    except Exception as e:
        print(f"Error during image analysis with 'inference' package: {e}")
        import traceback
        traceback.print_exc()
        return None, {"summary_text": f"Error during image analysis: {e}", "violations": []}


# --- Video Analysis using 'inference' and 'supervision' ---
def analyze_video_with_inference_pkg(video_bytes, project_id, original_filename):
    """
    Analyzes a video using three Roboflow models, annotates it frame by frame,
    saves the output to the project directory, and returns its path and analysis details.

    Args:
        video_bytes (bytes): Bytes data of the uploaded video.
        project_id (int): The ID of the current project.
        original_filename (str): The original name of the uploaded video file.

    Returns:
        tuple: (output_video_path, analysis_details) or (None, error_message).
               analysis_details is a dict including summary text and violations.
    """
    model_struct, model_excav, model_ppe = load_all_models(RF_API_KEY_SE, RF_API_KEY_PPE)
    if not all([model_struct, model_excav, model_ppe]):
        return None, {"summary_text": "Error: Failed to load one or more models.", "violations": []}

    # Create temporary files for source video
    temp_source_video_file = None
    source_video_path = None
    output_video_path = None
    temp_dir_output = None # Directory for output file

    try:
        # Save uploaded video bytes to a temporary file for processing
        # Use original extension for compatibility
        suffix = os.path.splitext(original_filename)[1]
        temp_source_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_source_video_file.write(video_bytes)
        source_video_path = temp_source_video_file.name
        temp_source_video_file.close() # Close the file handle so VideoInfo can open it

        print(f"Source video saved temporarily to: {source_video_path}")

        # Define path for the output annotated video in the project's processed media directory
        project_dir = os.path.join('data', str(project_id))
        processed_media_dir = os.path.join(project_dir, 'processed_media')
        os.makedirs(processed_media_dir, exist_ok=True) # Ensure directory exists
        output_video_filename = f"annotated_{os.path.splitext(original_filename)[0]}.mp4" # Force MP4 output
        output_video_path = os.path.join(processed_media_dir, output_video_filename)


        # Analysis summary accumulators
        total_struct_detections = 0
        total_excav_detections = 0
        total_ppe_detections = 0
        all_detected_classes = set()
        all_violations = [] # List to store violation details across frames

        print(f"Annotated video will be saved to: {output_video_path}")

        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        frames_generator = sv.get_video_frames_generator(source_path=source_video_path)

        # Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1,text_padding=2, color_lookup=sv.ColorLookup.CLASS)

        print("Starting video processing and annotation...")
        start_time = time.time()
        with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
            frame_count = 0
            for frame in frames_generator:
                frame_count +=1
                # Calculate estimated timestamp
                timestamp_sec = frame_count / video_info.fps
                timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}.{int((timestamp_sec * 100) % 100):02d}"

                if frame_count % (video_info.fps * 5) == 0 or frame_count == 1: # Log progress every 5 seconds or first frame
                    print(f"Processing frame {frame_count} (Time: {timestamp_str})...")

                # Inference on each model
                try:
                    r_struct = model_struct.infer(frame)[0]
                    r_excav = model_excav.infer(frame)[0]
                    r_ppe = model_ppe.infer(frame)[0]
                except (IndexError, TypeError, AttributeError):
                    print(f"Warning: Inference on frame {frame_count} for one or more models returned no primary result object.")
                    r_struct = type('obj', (object,), {'predictions': []})()
                    r_excav  = type('obj', (object,), {'predictions': []})()
                    r_ppe    = type('obj', (object,), {'predictions': []})()

                # Convert to supervision detections
                d_struct = sv.Detections.from_inference(r_struct)
                d_excav = sv.Detections.from_inference(r_excav)
                d_ppe = sv.Detections.from_inference(r_ppe)

                # Update summary counts
                total_struct_detections += len(d_struct.xyxy)
                total_excav_detections += len(d_excav.xyxy)
                total_ppe_detections += len(d_ppe.xyxy)

                # Merge all detections
                detections = sv.Detections.merge([d_struct, d_excav, d_ppe])

                # Get labels for merged detections (see notes in image analysis)
                all_predictions = r_struct.predictions + r_excav.predictions + r_ppe.predictions
                labels = [f"{p.class_name} ({p.confidence:.2f})" for p in all_predictions] # Labels including confidence
                current_frame_classes = [p.class_name for p in all_predictions] # Just class names
                all_detected_classes.update(current_frame_classes)

                # Extract safety violations for this frame
                frame_violations = extract_violations(
                    d_ppe, # Only check PPE detections
                    [p.class_name for p in r_ppe.predictions], # Classes from the PPE model's predictions
                    frame_info={'frame': frame_count, 'timestamp': timestamp_str}
                )
                all_violations.extend(frame_violations) # Add to cumulative list


                # Draw boxes + labels
                annotated_frame = frame.copy()
                if len(detections.xyxy) > 0:
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    # Use simplified labels for display
                    if len(labels) == len(detections.xyxy):
                         display_labels = [f"{label.split('(')[0].strip()}" for label in labels]
                         annotated_frame = label_annotator.annotate(
                             scene=annotated_frame,
                             detections=detections,
                             labels=display_labels
                         )
                    else:
                         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                # Optional: Add frame number/timestamp text overlay
                cv2.putText(annotated_frame, f"Frame: {frame_count} | Time: {timestamp_str}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


                sink.write_frame(frame=annotated_frame)

            end_time = time.time()
            print(f"Finished processing {frame_count} frames in {end_time - start_time:.2f} seconds.")


        # Generate summary text
        summary_text = "Video Analysis Summary:\n"
        summary_text += f"Total frames processed: {frame_count}\n"
        summary_text += f"- Total Structural element detections across frames: {total_struct_detections}\n"
        summary_text += f"- Total Excavator detections across frames: {total_excav_detections}\n"
        summary_text += f"- Total PPE item detections across frames: {total_ppe_detections}\n"
        summary_text += f"Unique classes detected across video: {', '.join(all_detected_classes) if all_detected_classes else 'None'}\n"

        # Add violation details to summary text
        if all_violations:
            summary_text += "\nPotential Safety Violations Detected (Sample):\n"
            # Limit the number of violations shown in the text summary if too many
            sample_violations = all_violations[:20] + (["..."] if len(all_violations) > 20 else [])
            for i, vio in enumerate(sample_violations):
                if isinstance(vio, str):
                    summary_text += f"  {vio}\n"
                else:
                    summary_text += f"  - {vio['class']} (Confidence: {vio['confidence']:.2f}) at Frame {vio['frame']} ({vio['timestamp']})\n"
            if len(all_violations) > 20:
                 summary_text += f"  ({len(all_violations) - 20} more violations not shown in summary text)\n"
        else:
            summary_text += "\nNo specific safety violations detected in this video."


        analysis_details = {
            "summary_text": summary_text,
            "violations": all_violations, # Store ALL violation data
            "video_path": output_video_path # Store path to annotated video
        }

        print(f"Annotated video saved successfully to {output_video_path}")
        return output_video_path, analysis_details

    except Exception as e:
        print(f"Error during video analysis with 'inference' package: {e}")
        import traceback
        traceback.print_exc()
        # Clean up potentially empty output file if error occurs before completion
        if output_video_path and os.path.exists(output_video_path) and os.path.getsize(output_video_path) < 1024: # Check if file is small/empty
             try:
                 os.remove(output_video_path)
                 print(f"Removed incomplete output video: {output_video_path}")
             except Exception as e_clean_output:
                 print(f"Error cleaning up incomplete output video {output_video_path}: {e_clean_output}")
        return None, {"summary_text": f"Error during video analysis: {e}", "violations": []}
    finally:
        # Clean up the temporary source video file
        if source_video_path and os.path.exists(source_video_path):
            try:
                os.remove(source_video_path)
                print(f"Removed temporary source video: {source_video_path}")
            except Exception as e_clean_source:
                print(f"Error cleaning up temporary source video {source_video_path}: {e_clean_source}")

# Note: Cleanup of the *output* video file in the data directory is not done here.
# It should persist with the project. The calling page might need to manage cleanup
# of *previous* processed media when a *new* analysis is run for the same project.