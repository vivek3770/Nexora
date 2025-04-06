import streamlit as st
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import torch
import traceback

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="AI Innovation Generator",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================
# Load AI Models
# ========================
def load_models():
    """
    Load text generation and image processing models with debug logs.
    Includes automatic fallback to CPU if GPU is unavailable.
    """
    try:
        print("Starting to load models...")
        
        # Load text generation model (Fallback to CPU if no GPU available)
        print("Loading text generation model (gpt2 for reliability)...")
        text_gen = pipeline(
            "text-generation",
            model="gpt2",  # Using GPT-2 for lightweight operation
            device=0 if torch.cuda.is_available() else -1
        )
        print("Text generation model loaded successfully.")
        
        # Load CLIP model and processor
        print("Loading CLIP model and processor...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model and processor loaded successfully.")
        
        print("All models loaded successfully.")
        return text_gen, clip_model, clip_processor
    except Exception as e:
        print(f"Error loading models: {traceback.format_exc()}")
        st.error(f"Error loading models: {e}")
        st.stop()

# Load models with debugging
text_generator, clip_model, clip_processor = load_models()

# ========================
# Core Functionalities
# ========================

def generate_innovation(prompt, context="general"):
    """
    Generate a unique, relevant, and practical solution without repeating or echoing the prompt.
    Ensures solutions are based on real-world applications and innovations.
    """
    try:
        # Refining the structured prompt
        structured_prompt = (
            f"Context: {context.capitalize()}.\n"
            "Based on the problem described below, provide a unique, detailed, and practical solution.\n"
            f"Problem: {prompt}\n"
            "Do not restate the problem. Focus exclusively on presenting the solution in a clear and actionable format."
        )

        # Generating output using the model
        outputs = text_generator(
            structured_prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,  # Encourages uniqueness
            top_p=0.9  # Balances creativity and practicality
        )

        # Extracting and refining the solution
        generated_text = outputs[0]["generated_text"].strip()

        # Removing the problem description and redundant phrases
        solution_start = generated_text.lower().find("solution:")
        if solution_start != -1:
            solution = generated_text[solution_start + len("solution:"):].strip()
        else:
            solution = generated_text.strip()

        # Ensuring the prompt is not echoed in the solution
        solution = solution.replace(prompt, "").strip()

        # Returning clean and concise output
        return solution if solution else "No solution could be generated."
    except Exception as e:
        st.error(f"Error generating innovation idea: {traceback.format_exc()}")
        return None



def analyze_data(df):
    """
    Analyze the uploaded CSV file to derive actionable insights and potential ideas.
    """
    try:
        summary = f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**."
        insights = f"Key columns: {', '.join(df.columns)}"
        stats = df.describe().to_dict()
        use_case = (
            "Based on these insights, predictive analytics models, trend analysis, or dashboards "
            "can be created to drive business or operational improvements."
        )
        return summary, insights, stats, use_case
    except Exception as e:
        st.error(f"Error analyzing data: {traceback.format_exc()}")
        return None, None, None, None

def generate_random_concept():
    """
    Generate a completely random innovation concept with a use case.
    """
    concepts = [
        {
            "idea": "AI-powered IoT sensors to optimize energy usage in smart homes.",
            "use_case": "Reduces energy bills by up to 40% while ensuring sustainability."
        },
        {
            "idea": "AR glasses for real-time translation during conversations.",
            "use_case": "Breaks language barriers, enabling seamless global communication."
        },
        {
            "idea": "Wearable devices that monitor mental health and provide guided therapy sessions.",
            "use_case": "Helps individuals manage stress and anxiety proactively."
        },
        {
            "idea": "Personalized e-learning platforms using adaptive AI to match individual learning styles.",
            "use_case": "Improves student outcomes and engagement by 50%."
        },
        {
            "idea": "Blockchain-based food traceability system.",
            "use_case": "Ensures food safety by tracking the origin and quality of produce."
        }
    ]
    return random.choice(concepts)

# ========================
# Streamlit User Interface
# ========================
st.title("🚀 AI Innovation Generator")
st.markdown("Unleash the power of **AI-generated solutions** for real-world problems.")

# Sidebar for input selection and context
input_type = st.sidebar.radio(
    "Select the type of input:",
    ["Text", "Image", "CSV Data", "Random Concept"],
    index=0
)
context = st.sidebar.selectbox(
    "Choose the context for innovations:",
    ["General", "Healthcare", "Education", "Sustainability", "Robotics", "Agriculture", "Technology"],
    index=0
)

# Handle inputs based on user selection
if input_type == "Text":
    user_input = st.text_area("Describe your problem or idea:")
    if st.button("Generate Innovation"):
        if user_input.strip():
            with st.spinner("Generating innovative solution..."):
                output = generate_innovation(user_input, context)
                if output:
                    st.success("🎯 Innovative Solution:")
                    st.write(output)
        else:
            st.warning("Please enter a valid problem statement or idea.")

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image & Generate Idea"):
            with st.spinner("Analyzing image..."):
                concepts = analyze_image(image)
                if concepts:
                    st.success("🔍 Top Relevant Concepts:")
                    for label, score in concepts:
                        st.write(f"- **{label.capitalize()}** — Probability: {score:.4f}")
                    top_concept = concepts[0][0]
                    output = generate_innovation(f"{top_concept} innovation", context)
                    if output:
                        st.markdown("✨ **Generated Idea:**")
                        st.write(output)

elif input_type == "CSV Data":
    uploaded_csv = st.file_uploader("Upload a CSV file:", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df)
        if st.button("Generate Insights & Innovation"):
            with st.spinner("Analyzing data..."):
                summary, insights, stats, use_case = analyze_data(df)
                if summary and insights and stats and use_case:
                    st.success("📊 Data Insights:")
                    st.write(summary)
                    st.write(insights)
                    st.json(stats)
                    st.markdown("💡 **Use Case:**")
                    st.write(use_case)

elif input_type == "Random Concept":
    if st.button("Generate Random Concept"):
        concept = generate_random_concept()
        st.markdown("🎲 **Random Concept:**")
        st.write(f"💡 Idea: {concept['idea']}")
        st.write(f"🌍 Use Case: {concept['use_case']}")

st.write("---")
st.caption("Powered by Hugging Face Transformers, CLIP, and Streamlit. Built for meaningful innovations!")
