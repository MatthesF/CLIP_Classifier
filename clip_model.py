from transformers import CLIPProcessor, CLIPModel

def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

def get_text_embeddings(processor, model, texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)
    return text_features