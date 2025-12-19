"""
Streamlit Web Demo for Face Verification using MagFace
Run: streamlit run demo.py
"""
import streamlit as st
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import tempfile

import config
from models import create_magface_model


@st.cache_resource
def load_model(checkpoint_path=None):
    """Load MagFace model (cached)"""
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.MODEL_ROOT, 'magface_best_model.pth')
    
    model = create_magface_model(pretrained=True, checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image):
    """
    Preprocess PIL image for model input
    
    Args:
        image: PIL Image
        
    Returns:
        tensor: Preprocessed image tensor
    """
    # Convert PIL to numpy array
    img = np.array(image)
    
    # Convert RGB to BGR if needed (for consistency)
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 112x112
    img = cv2.resize(img, (112, 112))
    
    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img)
    return img_tensor


def extract_feature(model, device, image):
    """Extract feature embedding from image"""
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = model.extract_features(img_tensor)
        
        # Handle MagFace returning (embeddings, magnitudes)
        if isinstance(feature, tuple):
            feature = feature[0]
        
        # Normalize feature
        feature = torch.nn.functional.normalize(feature, p=2, dim=1)
    
    return feature


def compute_similarity(feature1, feature2):
    """Compute cosine similarity between two features"""
    similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
    return similarity.item()


def main():
    # Page config
    st.set_page_config(
        page_title="Face Verification - MagFace",
        page_icon="👤",
        layout="wide"
    )
    
    # Title
    st.title("👤 Face Verification System")
    st.markdown("### Using MagFace Deep Learning Model")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Threshold for determining if two faces belong to the same person"
        )
        
        st.markdown("---")
        st.markdown("### 📖 How to use")
        st.markdown("""
        1. Upload two face images
        2. Click 'Verify Faces'
        3. View similarity score and result
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Model:** MagFace  
        **Device:** {config.DEVICE}  
        **Threshold:** {threshold}
        """)
    
    # Load model
    with st.spinner("Loading MagFace model..."):
        try:
            model, device = load_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.stop()
    
    # Main content - Two columns for image upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Image 1")
        uploaded_file1 = st.file_uploader(
            "Upload first face image",
            type=['jpg', 'jpeg', 'png'],
            key="image1"
        )
        
        if uploaded_file1 is not None:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Image 1", use_column_width=True)
    
    with col2:
        st.subheader("📷 Image 2")
        uploaded_file2 = st.file_uploader(
            "Upload second face image",
            type=['jpg', 'jpeg', 'png'],
            key="image2"
        )
        
        if uploaded_file2 is not None:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Image 2", use_column_width=True)
    
    st.markdown("---")
    
    # Verify button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        verify_button = st.button("🔍 Verify Faces", type="primary", use_container_width=True)
    
    # Verification
    if verify_button:
        if uploaded_file1 is None or uploaded_file2 is None:
            st.warning("⚠️ Please upload both images first!")
        else:
            with st.spinner("Processing images..."):
                try:
                    # Extract features
                    feature1 = extract_feature(model, device, image1)
                    feature2 = extract_feature(model, device, image2)
                    
                    # Compute similarity
                    similarity = compute_similarity(feature1, feature2)
                    
                    # Determine if same person
                    is_same_person = similarity >= threshold
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Verification Results")
                    
                    # Metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Similarity Score",
                            value=f"{similarity:.4f}",
                            delta=f"{(similarity - threshold):.4f}" if similarity >= threshold else f"{(similarity - threshold):.4f}"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Threshold",
                            value=f"{threshold:.4f}"
                        )
                    
                    with metric_col3:
                        confidence = similarity * 100
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.2f}%"
                        )
                    
                    # Result display
                    st.markdown("---")
                    
                    if is_same_person:
                        st.success("### ✅ SAME PERSON")
                        st.balloons()
                        st.markdown(f"""
                        The two images appear to show the **same person**.
                        
                        - **Similarity Score:** {similarity:.4f}
                        - **Threshold:** {threshold:.4f}
                        - **Confidence:** {confidence:.2f}%
                        """)
                    else:
                        st.error("### ❌ DIFFERENT PEOPLE")
                        st.markdown(f"""
                        The two images appear to show **different people**.
                        
                        - **Similarity Score:** {similarity:.4f}
                        - **Threshold:** {threshold:.4f}
                        - **Confidence:** {(1-similarity)*100:.2f}% (different)
                        """)
                    
                    # Progress bar visualization
                    st.markdown("### 📈 Similarity Visualization")
                    st.progress(min(similarity, 1.0))
                    
                    # Detailed info
                    with st.expander("🔬 Technical Details"):
                        st.markdown(f"""
                        **Feature Extraction:**
                        - Image 1 feature shape: {feature1.shape}
                        - Image 2 feature shape: {feature2.shape}
                        
                        **Similarity Computation:**
                        - Method: Cosine Similarity
                        - Score Range: [0, 1]
                        - Current Score: {similarity:.6f}
                        
                        **Decision:**
                        - Threshold: {threshold}
                        - Result: {"Same Person" if is_same_person else "Different People"}
                        """)
                
                except Exception as e:
                    st.error(f"❌ Error during verification: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Face Verification System using MagFace | CS331 Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
