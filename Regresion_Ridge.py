import streamlit as st
pip install pycaret
from pycaret.regression import load_model
from pycaret.regression import predict_model
import pandas as pd

# Cargar el modelo entrenado
modelo_entrenado = load_model('mi_modelo')

# Título de la aplicación
st.title('App de Predicción con PyCaret (Regresión)')

# Cargar el archivo CSV de entrada
archivo_cargado = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_cargado is not None:
    # Leer el archivo CSV
    dataset = pd.read_csv(archivo_cargado)
    
    # Mostrar el dataset cargado
    st.write("Dataset cargado:")
    st.write(dataset)
    
    # Hacer predicciones
    predicciones = predict_model(modelo_entrenado, data=dataset)
    
    # Mostrar las predicciones
    st.write("Predicciones:")
    st.write(predicciones)

    # Opción para descargar el resultado
    st.download_button(label="Descargar Predicciones",
                       data=predicciones.to_csv(index=False),
                       file_name='predicciones.csv',
                       mime='text/csv')
