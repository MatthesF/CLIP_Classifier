import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clip_model import load_clip_model, get_text_embeddings
from utils import cosine_similarity, softmax_with_temperature, process_texts

# Load CSS for custom styling
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def image_classification(image, vector_dictionary):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    emb = model.get_image_features(**inputs)
    cosine_sim = cosine_similarity(emb.detach().numpy(), vector_dictionary_arr)
    
    # Apply temperature-scaled softmax to cosine similarity scores
    softmax_scores = softmax_with_temperature(cosine_sim, temperature=.075)
    
    nearest_word = list(vector_dictionary.keys())[np.argmax(softmax_scores)]
    return nearest_word,softmax_scores

if __name__ == "__main__":
    load_css()
    st.title("🚀 Image Classification with CLIP")
    
    # Load the model and processor
    processor, model = load_clip_model()

    # Allow users to select or add objects for classification
    st.sidebar.header("🔧 Settings")
    default_objects = ["dog", "cat", "flower", "car", "plane", "house", "book", "phone"]
    selected_objects = st.sidebar.multiselect("Select objects", default_objects, default=default_objects)
    custom_object = st.sidebar.text_input("Add a custom object")
    if custom_object:
        selected_objects.append(default_objects)

    # Prepare text embeddings
    texts = process_texts(selected_objects)
    vector_dictionary = get_text_embeddings(processor, model, texts)
    vector_dictionary_arr = vector_dictionary.detach().numpy().copy()
    vector_dictionary = {t.split()[-1]: v for t, v in zip(texts, vector_dictionary)}

    # Streamlit UI for image upload
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            label, similarity_scores = image_classification(image, vector_dictionary)

            st.write(f"### Predicted Label: **{label.capitalize()}**")
            st.write("### Similarity Scores:")
            data = {obj:float(score) for obj,score in zip(selected_objects[:-1],similarity_scores)}
            df = pd.DataFrame(list(data.items()), columns=['Category', 'Value']).set_index('Category')
            st.bar_chart(df)

			




