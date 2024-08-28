from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import streamlit as st

def cosine_similarity(A, B):
	"""
	function to calculate cosine similarity between 1D array and 2D array
	A: 1D array
	B: 2D array"""
	return np.array([np.dot(A, b) / (np.linalg.norm(A) * np.linalg.norm(b)) for b in B])

def image_classification(image, vector_dictionary):
	inputs = processor(images=image, return_tensors="pt", padding=True)
	emb = model.get_image_features(**inputs)
	cosine_sim = cosine_similarity(emb.detach().numpy(), vector_dictionary_arr)
	#for i, (k, v) in enumerate(vector_dictionary.items()):
	#	print(f"{image_path} and {k} similarity: {cosine_sim[i]}")
	nearest_word = list(vector_dictionary.keys())[np.argmax(cosine_sim)]
	return nearest_word

# import models
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

texts = [f"a photo of a {obj}" for obj in ["dog", "cat", "flower","car","plane","house", "book", "phone","kid","man","woman"]]
inputs = processor(text=texts, return_tensors="pt", padding=True)
vector_dictionary = model.get_text_features(**inputs)
vector_dictionary_arr = vector_dictionary.detach().numpy().copy()
vector_dictionary = {t.split()[-1]: v for t, v in zip(texts, vector_dictionary)}


if __name__ == "__main__":
	"# Image Classification with CLIP"
	# make image to vector with CLIP
	uploaded_file = st.file_uploader(
		"Choose a Image file", accept_multiple_files=False, type="png"
	)
	if uploaded_file is not None:
		image = Image.open(uploaded_file)
		st.image(image)
		st.write(image_classification(image, vector_dictionary))


