import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.title("Clasificación de Residuos Reciclables")
st.write("Suba una imagen para clasificarla con el modelo entrenado.")

@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model('modelo_reciclaje.h5')
    with open('clases.json', 'r') as f:
        clases = json.load(f)
    return modelo, clases

modelo, clases = cargar_modelo()

archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert('RGB')
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    img_resized = imagen.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)
    clase_predicha = clases[np.argmax(prediccion)]
    confianza = np.max(prediccion) * 100

    st.success(f"Clasificación: **{clase_predicha}**")
    st.write(f"Confianza: {confianza:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")