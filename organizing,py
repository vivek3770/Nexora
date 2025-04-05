import streamlit as st

# Set Streamlit page config FIRST
st.set_page_config(page_title="Open-Source AI Innovator", layout="centered")

from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import torch
import random

# =======================
# Load Open-Source Models
# =======================
@st.cache_resource
def load_models():
    text_gen = pipeline("text-generation", model="gpt2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return text_gen, clip_model, clip_processor

text_generator, clip_model, clip_processor = load_models()

# =======================
# Core Functionality
# =======================
def generate_innovation(prompt):
    outputs = text_generator(prompt, max_length=100, num_return_sequences=1)
    return outputs[0]['generated_text'].strip()

def analyze_image(image):
    # Predefined concept texts (for basic similarity)
    concepts = [
        "healthcare", "education", "business", "robotics",
        "environment", "remote work", "wearable tech",
        "gaming", "AI assistant", "smart agriculture"
    ]
    inputs = clip_processor(text=concepts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, len(concepts)]
        probs = logits_per_image.softmax(dim=1).squeeze().tolist()
        concept_scores = list(zip(concepts, probs))
        concept_scores.sort(key=lambda x: x[1], reverse=True)
    return concept_scores[:5]  # return top 5 matching concepts

def analyze_data(df):
    summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    stats = df.describe().to_dict()
    return summary, stats

def random_concept():
    concepts = [
        "🧠 AI therapist that adapts to patient emotions in real-time.",
        "🎓 Education assistant that personalizes content daily.",
        "💼 Business analyzer that finds gaps in your market strategy.",
        "🎮 Game generator that builds storylines based on your day.",
        "🏥 Early-detection health assistant using wearable data."
    ]
    return random.choice(concepts)

# =======================
# Streamlit UI
# =======================
st.title("🚀 Open-Source AI Innovator")
st.markdown("Upload **text**, **images**, or **data** to generate innovative solutions using open-source models!")

input_type = st.radio("Choose your input type:", ["Text", "Image", "CSV Data", "Random Innovation"])

if input_type == "Text":
    user_input = st.text_area("Describe your problem or idea (e.g. healthcare, education, etc.):")
    if st.button("Generate Innovation"):
        if user_input.strip():
            with st.spinner("Generating..."):
                output = generate_innovation(f"Generate a new innovation idea for: {user_input}")
                st.success("✨ Innovative Idea:")
                st.write(output)
        else:
            st.warning("Please enter some input.")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Innovation from Image"):
            with st.spinner("Analyzing Image..."):
                concepts = analyze_image(image)
                st.success("💡 Top Matching Concepts:")
                for label, score in concepts:
                    st.write(f"**{label.capitalize()}** — Score: {score:.4f}")
                top_prompt = concepts[0][0]
                st.markdown("🧠 Generating idea using top concept...")
                output = generate_innovation(f"Generate a new innovation idea in {top_prompt}")
                st.markdown("✨ **Innovative Idea:**")
                st.write(output)

elif input_type == "CSV Data":
    uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df)
        if st.button("Generate Innovation from Data"):
            with st.spinner("Analyzing Data..."):
                summary, stats = analyze_data(df)
                st.success("📊 Data Summary:")
                st.write(summary)
                st.json(stats)
                st.markdown("💡 Now use this summary as a prompt in the Text section to generate innovation.")

elif input_type == "Random Innovation":
    if st.button("Surprise Me!"):
        st.write("🎲 Here's a random innovative concept:")
        st.markdown(f"**{random_concept()}**")

st.write("---")
st.caption("Built using Hugging Face Transformers, CLIP, and Streamlit — by Piyush 🚀")