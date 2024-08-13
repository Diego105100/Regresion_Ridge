import streamlit as st
from pycaret.regression import load_model, predict_model
import pandas as pd

# Cargar el modelo entrenado
try:
    modelo_entrenado = load_model('mi_modelo')
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Título de la aplicación
st.title('App de Predicción con PyCaret (Regresión)')

# Cargar el archivo CSV de entrada
archivo_cargado = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_cargado is not None:
    # Leer el archivo CSV
    try:
        dataset = pd.read_csv(archivo_cargado)
        st.write("Dataset cargado:")
        st.write(dataset)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")

    if 'dataset' in locals():  # Verifica si el dataset se cargó correctamente
        # Validar las columnas requeridas (ajusta según las columnas que necesita tu modelo)
        columnas_requeridas = ['variable1', 'variable2', 'variable3']  # Reemplaza con las columnas reales
        if set(columnas_requeridas).issubset(set(dataset.columns)):
            # Hacer predicciones
            try:
                predicciones = predict_model(modelo_entrenado, data=dataset)
                st.write("Predicciones:")
                st.write(predicciones)
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

            # Opción para descargar el resultado
            st.download_button(label="Descargar Predicciones",
                               data=predicciones.to_csv(index=False),
                               file_name='predicciones.csv',
                               mime='text/csv')
        else:
            st.error("El archivo CSV no tiene las columnas requeridas.")
