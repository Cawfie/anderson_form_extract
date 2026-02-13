
# How to Run:
# 1) Create a Virtual Ennironment 
# 2) Install requirements: pip install -r requirements.txt
# 3) Add your API keys to config.toml file
# 4) Run the app: streamlit run gemini.py

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import streamlit as st   # Using Streamlit for interface
from google import genai
from google.genai import types
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor
import boto3
from io import BytesIO
from datetime import datetime
from pathlib import Path
from prompts import PRECISION_PROMPT, PERSONAL_DETAILS_PROMPT  # Importing your prompts

# Load configuration from TOML file
config = st.secrets

# --- CONFIGURATION ---
GEMINI_API_KEY = config["gemini"]["GEMINI_API_KEY"]
MODEL_ID = "gemini-3-pro-preview"  # <----------- Model for checkbox extraction
MODEL_FLASH_ID = "gemini-2.5-flash"  # <----------- Model for personal details

# --- S3 CONFIGURATION ---
AWS_ACCESS_KEY = config["s3"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = config["s3"]["AWS_SECRET_KEY"]
S3_BUCKET_NAME = config["s3"]["S3_BUCKET_NAME"]
S3_REGION = config["s3"].get("S3_REGION", "us-east-1")

st.set_page_config(page_title="Medical OCR", layout="wide")
st.title("Medical Checkbox Extractor")

# Initialize session state for tracking extraction
if 'extraction_in_progress' not in st.session_state:
    st.session_state.extraction_in_progress = False
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None

# Initialize Clients
client = genai.Client(api_key=GEMINI_API_KEY)
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)

# --- S3 HELPER FUNCTIONS ---
def ensure_json_folder_exists():
    """Check if 'json' folder exists in S3, create if not"""
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key="json/")
    except:
        # Folder doesn't exist, create it by uploading an empty object
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key="json/")

def list_s3_images():
    """List all image files from S3 bucket root (excluding json/ folder)"""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        if 'Contents' in response:
            # Filter only image files that are not in the json/ folder
            images = [obj['Key'] for obj in response['Contents'] 
                     if obj['Key'].endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')) 
                     and not obj['Key'].startswith('json/')]
            return sorted(images)
        return []
    except Exception as e:
        st.error(f"Error listing S3 objects: {e}")
        return []

def download_image_from_s3(s3_key):
    """Download image from S3 bucket"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data))
    except Exception as e:
        st.error(f"Error downloading image from S3: {e}")
        return None

def upload_json_to_s3(merged_data, original_filename):
    """Upload merged JSON results to S3 in json/ folder with patient name"""
    try:
        ensure_json_folder_exists()
        
        # Extract patient name from personal details
        patient_name = merged_data.get('personal_details', {}).get('Name', 'unknown')
        # Sanitize patient name for filename (replace spaces and special chars)
        safe_patient_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-')).strip()
        
        # Create JSON filename based on patient name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{safe_patient_name}_{timestamp}.json"
        s3_key = f"json/{json_filename}"
        
        # Upload JSON to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(merged_data, indent=2),
            ContentType='application/json'
        )
        
        return s3_key, timestamp
    except Exception as e:
        st.error(f"Error uploading JSON to S3: {e}")
        return None, None

def list_s3_json_files():
    """List all JSON files from json/ folder in S3"""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="json/")
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
            return sorted(files, reverse=True)  # Sort by newest first
        return []
    except Exception as e:
        st.error(f"Error listing JSON files from S3: {e}")
        return []

def download_json_from_s3(s3_key):
    """Download and parse JSON file from S3"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        json_data = json.loads(response['Body'].read().decode('utf-8'))
        return json_data
    except Exception as e:
        st.error(f"Error downloading JSON from S3: {e}")
        return None

st.divider()

# Main layout: left = image + extract, right = JSON viewer
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📸 Select Image")
    s3_images = list_s3_images()
    if s3_images:
        selected_image = st.selectbox("Select image from S3:", s3_images, key="extract_image_select")
        img = download_image_from_s3(selected_image)
        filename = selected_image.split('/')[-1]
        if img:
            st.image(img, caption="Original Document", use_container_width=True)
        else:
            img = None
            filename = None
    else:
        st.warning("No images found in S3 bucket")
        img = None
        filename = None

    # Extraction button
    extract_disabled = img is None or st.session_state.extraction_in_progress
    if st.button("🔄 Extract Data", disabled=extract_disabled, key="extract_btn", use_container_width=True):
        st.session_state.extraction_in_progress = True
        try:
            with st.spinner("⏳ Extracting Checkbox Data..."):
                def extract_checkbox_data():
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=[PRECISION_PROMPT, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    return json.loads(response.text)

                checkbox_data = extract_checkbox_data()
                st.success("✓ Checkbox Data Extracted")

            with st.spinner("⏳ Extracting Personal Details..."):
                def extract_personal_data():
                    response = client.models.generate_content(
                        model=MODEL_FLASH_ID,
                        contents=[PERSONAL_DETAILS_PROMPT, img],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )
                    return json.loads(response.text)

                personal_data = extract_personal_data()
                st.success("✓ Personal Details Extracted")

            with st.spinner("⏳ Uploading to S3..."):
                merged_data = {
                    "personal_details": personal_data,
                    "medical_prescriptions": checkbox_data
                }
                s3_key, timestamp = upload_json_to_s3(merged_data, filename)
                if s3_key:
                    st.success("✓ Uploaded to S3")
                else:
                    st.warning("⚠ Upload failed")

            # Save result to session so right column can display it
            st.session_state.extraction_result = merged_data
            st.session_state.extraction_s3_key = s3_key
            st.session_state.extraction_timestamp = timestamp

        except Exception as e:
            st.error(f"Extraction error: {e}")
        finally:
            st.session_state.extraction_in_progress = False

        # Show latest extraction result below the extract button (left pane)
        if st.session_state.get('extraction_result'):
            st.divider()
            st.subheader("📊 Extracted Data")
            st.json(st.session_state['extraction_result'])
            if st.session_state.get('extraction_s3_key'):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"📁 S3 Path: `{st.session_state.get('extraction_s3_key')}`")
                with col_b:
                    st.info(f"🕐 Timestamp: {st.session_state.get('extraction_timestamp')}")

with right_col:
    st.subheader("📂 Saved JSON Files (S3)")
    json_files = list_s3_json_files()
    if json_files:
        selected_json = st.selectbox(
            "Select a file:",
            json_files,
            format_func=lambda x: x.replace('json/', '').replace('.json', ''),
            key="view_json_select"
        )
        json_data = download_json_from_s3(selected_json)
        if json_data:
            st.success(f"✓ Loaded: {selected_json.replace('json/', '')}")
            st.json(json_data)
    else:
        st.info("No JSON files found in S3.")