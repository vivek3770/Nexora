import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import CLIPProcessor, CLIPModel, BlipForConditionalGeneration, BlipProcessor
import torch
from PIL import Image
import pandas as pd
import re
import os
os.environ["STREAMLIT_WATCHER_DISABLE_AUTO_RELOAD"] = "true"
import time
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Configuration
st.set_page_config(page_title="InnovateAI Pro", layout="wide", page_icon="🚀")

# =======================
# Enhanced Model Loading
# =======================
@st.cache_resource(show_spinner="Loading AI Engine...")
def load_models():
    models = {}

    try:
        # Use GPT-2 instead of Falcon for CPU compatibility
        models['text_gen'] = pipeline(
            "text-generation",
            model="gpt2",
            tokenizer="gpt2"
        )

        # Image Understanding
        models['clip_model'], models['clip_processor'] = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        ), CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        # Advanced Image Captioning
        models['blip_processor'], models['blip_model'] = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ), BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )

        logging.info("Models loaded successfully.")
        return models

    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return None


# Initialize models once
models = load_models()

if models is None:
    st.error("Failed to load models.")
else:
    def generate_innovation(prompt, context=""):
        try:
            few_shot_example = """
You are an expert innovation strategist. Your task is to suggest actionable, innovative solutions using cutting-edge technologies.

---
Input: How can AI help in disaster management?
Context: Natural disasters are becoming more frequent and destructive.

Answer:
1. Problem: Disaster response is often delayed and uncoordinated.
2. Insight: Real-time AI analysis of satellite imagery can guide rescue efforts.
3. Solution: A prediction platform combining AI and satellite data to trigger alerts and dispatch drones.
4. Impact: Can reduce rescue time by up to 70%.
5. First Steps:
   - Gather satellite disaster imagery datasets.
   - Train a classification model for damage detection.
   - Integrate model into an alerting dashboard.

---
Input: Suggest a blockchain-based solution for agriculture.
Context: Farmers struggle with price transparency and fair payments.

Answer:
1. Problem: Lack of trust and transparency in agricultural supply chains.
2. Insight: Blockchain can enable tamper-proof transactions and pricing data.
3. Solution: A decentralized farmer-to-market platform where all trades are recorded on-chain, ensuring fair payments.
4. Impact: Reduces middlemen exploitation, increases farmer income by 30%.
5. First Steps:
   - Identify core stakeholders and crop data schema.
   - Develop a smart contract for transactions.
   - Pilot with 100 farmers and local markets.

---
"""
            final_prompt = f"""{few_shot_example}
Input: {prompt}
Context: {context if context else 'N/A'}

Answer:"""

            response = models['text_gen'](
                final_prompt,
                max_new_tokens=300,
                temperature=0.5,
                top_p=0.8,
                repetition_penalty=1.3,
                do_sample=True
            )

            raw_text = response[0]['generated_text']
            return postprocess_output(raw_text, final_prompt, prompt)

        except Exception as e:
            logging.error(f"Error generating innovation: {e}")
            return f"🚨 Error: {str(e)}"

    def postprocess_output(text, final_prompt, original_prompt):
        try:
            logging.info(f"Full text before processing: {text}")

            generated_text = text.replace(final_prompt, "").strip()
            generated_text = generated_text.replace(original_prompt, "").strip()
            cleaned = generated_text.strip()
            logging.info(f"Cleaned text after removing prompts: {cleaned}")

            cleaned = re.sub(r'\n(\d+\.)', r'\n### \1', cleaned)
            cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
            cleaned = cleaned.replace('\n', '\n\n')
            cleaned = re.sub(r'http\S+', '', cleaned)  # remove URLs
            cleaned = re.split(r'\n---', cleaned)[0]   # stop after expected output


            logging.info(f"Final cleaned text: {cleaned}")

            return cleaned

        except Exception as e:
            logging.error(f"Error post-processing output: {e}")
            return "Error during post-processing. Check logs."

    def analyze_image(image):
        try:
            concepts = [
                "biomimicry", "nanotechnology", "circular economy",
                "generative design", "quantum computing", "neural interfaces",
                "synthetic biology", "swarm intelligence", "causal inference"
            ]

            inputs = models['clip_processor'](
                text=concepts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = models['clip_model'](**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

            return sorted(zip(concepts, probs), key=lambda x: -x[1])[:3]

        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            return f"🚨 Error: {str(e)}"

    def caption_image(image):
        try:
            inputs = models['blip_processor'](image, return_tensors="pt")
            with torch.no_grad():
                output = models['blip_model'].generate(**inputs, max_new_tokens=100)
            return models['blip_processor'].decode(output[0], skip_special_tokens=True)

        except Exception as e:
            logging.error(f"Error captioning image: {e}")
            return f"🚨 Error: {str(e)}"

    @st.cache_data(ttl=3600)
    def load_data(file):
        try:
            if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(file)
            else:
                return pd.read_csv(file)

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

    def analyze_data(df):
        try:
            insights = []
            for col in df.select_dtypes(include='number'):
                if df[col].nunique() > 5:
                    trend = "↑ Increasing" if df[col].iloc[-5:].mean() > df[col].mean() else "↓ Decreasing"
                    insights.append(f"{col}: {trend} trend (last 5 entries)")
            return insights

        except Exception as e:
            logging.error(f"Error analyzing data: {e}")
            return f"🚨 Error: {str(e)}"

    st.title("🧠 InnovateAI Pro")
    st.markdown("### Cross-Domain Innovation Generator")

    input_type = st.radio("Input Type:", ["Text", "Image", "Data", "Hybrid"], horizontal=True)

    if input_type == "Text":
        prompt = st.text_area("Describe your challenge or opportunity:",
                            placeholder="e.g., 'Sustainable packaging for tropical climates'")

        if st.button("Generate Solution"):
            with st.spinner("🚀 Connecting unexpected ideas..."):
                start_time = time.time()
                result = generate_innovation(prompt)
                end_time = time.time()
                st.write(f"Generated in {end_time - start_time:.2f} seconds")
            st.markdown("## 💡 Breakthrough Concept")
            if isinstance(result, str) and result.startswith("🚨 Error"):
                st.error(result)
            else:
                st.markdown(result)

    elif input_type == "Image":
        img = st.file_uploader("Upload Inspiration Image:", type=["png", "jpg", "jpeg"])
        if img:
            image = Image.open(img).convert('RGB')
            st.image(image, width=400)

            if st.button("Extract Innovation"):
                with st.spinner("🔍 Decoding visual patterns..."):
                    start_time = time.time()
                    caption = caption_image(image)
                    concepts = analyze_image(image)
                    context = f"Image shows: {caption}\nKey concepts: {', '.join([c[0] for c in concepts])}"
                    result = generate_innovation("Create innovation from image", context)
                    end_time = time.time()
                    st.write(f"Generated in {end_time - start_time:.2f} seconds")

                st.markdown("## 🎨 Vision-Based Innovation")
                st.markdown(result)

    elif input_type == "Data":
        data_file = st.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
        if data_file:
            df = load_data(data_file)
            if df is not None:
                st.dataframe(df.head(), use_container_width=True)

                if st.button("Generate Data-Driven Innovation"):
                    with st.spinner("📈 Mining hidden patterns..."):
                        start_time = time.time()
                        insights = analyze_data(df)
                        context = f"Data insights:\n- " + "\n- ".join(insights)
                        result = generate_innovation("Create innovation from data trends", context)
                        end_time = time.time()
                        st.write(f"Generated in {end_time - start_time:.2f} seconds")

                    st.markdown("## 📊 Data-Powered Solution")
                    st.markdown(result)
            else:
                st.error("Failed to load data.")

    elif input_type == "Hybrid":
        st.write("Hybrid input handling is under development.")

st.markdown("---")
st.caption("InnovateAI Pro v1.0 | Powered by Hugging Face Transformers — by Piyush 🚀")
