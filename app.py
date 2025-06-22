import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained generator
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_generator.h5")

generator = load_model()
noise_dim = 100

st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Pick a digit (0–9):", list(range(10)))
generate = st.button("Generate 5 Samples")

if generate:
    st.subheader(f"Generated images of digit {digit} (not conditional, just random GAN samples)")
    
    # Generate 5 random noise vectors
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        noise = tf.random.normal([1, noise_dim])
        img = generator(noise, training=False).numpy()[0, :, :, 0]
        axs[i].imshow(img * 127.5 + 127.5, cmap='gray')  # Rescale from [-1,1] to [0,255]
        axs[i].axis('off')

    st.pyplot(fig)
