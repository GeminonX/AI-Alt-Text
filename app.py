import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

#load model/processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

st.title("AI Generated Alt Text")
st.write("Upload an image and generate alt text optimized for accessibility")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Upload Image", use_column_width=True)

    if st.button("Generate AI Text"):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs, max_length=80)
        caption = processor.decode(output[0], skip_special_tokens=True)

        st.subheader("Generate Alt Text:")
        st.success(caption)

