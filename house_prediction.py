import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# List of all location names used during training
location_names = [
    'Abrantes, Madrid', 'Acacias, Madrid', 'Adelfas, Madrid', 'Alameda de Osuna, Madrid', 'Almagro, Madrid',
    'Almendrales, Madrid', 'Aluche, Madrid', 'Ambroz, Madrid', 'Apóstol Santiago, Madrid', 'Arapiles, Madrid',
    'Aravaca, Madrid', 'Arganzuela, Madrid', 'Argüelles, Madrid', 'Arroyo del Fresno, Madrid', 'Atalaya, Madrid',
    'Barajas, Madrid', 'Barrio de Salamanca, Madrid', 'Bellas Vistas, Madrid', 'Bernabéu-Hispanoamérica, Madrid',
    'Berruguete, Madrid', 'Buena Vista, Madrid', 'Butarque, Madrid', 'Campamento, Madrid',
    'Campo de las Naciones-Corralejos, Madrid', 'Canillas, Madrid', 'Carabanchel, Madrid', 'Casa de Campo, Madrid',
    'Casco Histórico de Barajas, Madrid', 'Casco Histórico de Vallecas, Madrid', 'Casco Histórico de Vicálvaro, Madrid',
    'Castellana, Madrid', 'Castilla, Madrid', 'Centro, Madrid', 'Chamartín, Madrid', 'Chamberí, Madrid',
    'Chopera, Madrid',
    'Chueca-Justicia, Madrid', 'Ciudad Jardín, Madrid', 'Ciudad Lineal, Madrid', 'Ciudad Universitaria, Madrid',
    'Colina, Madrid', 'Comillas, Madrid', 'Concepción, Madrid', 'Conde Orgaz-Piovera, Madrid', 'Costillares, Madrid',
    'Cuatro Caminos, Madrid', 'Cuatro Vientos, Madrid', 'Cuzco-Castillejos, Madrid', 'Delicias, Madrid',
    'El Cañaveral - Los Berrocales, Madrid', 'El Pardo, Madrid', 'El Plantío, Madrid', 'El Viso, Madrid',
    'Ensanche de Vallecas - La Gavia, Madrid', 'Entrevías, Madrid', 'Estrella, Madrid', 'Fontarrón, Madrid',
    'Fuencarral, Madrid', 'Fuente del Berro, Madrid', 'Fuentelarreina, Madrid', 'Gaztambide, Madrid', 'Goya, Madrid',
    'Guindalera, Madrid', 'Horcajo, Madrid', 'Hortaleza, Madrid', 'Huertas-Cortes, Madrid', 'Ibiza, Madrid',
    'Imperial, Madrid', 'Jerónimos, Madrid', 'La Paz, Madrid', 'Las Tablas, Madrid', 'Latina, Madrid',
    'Lavapiés-Embajadores, Madrid', 'Legazpi, Madrid', 'Lista, Madrid', 'Los Cármenes, Madrid', 'Los Rosales, Madrid',
    'Los Ángeles, Madrid', 'Lucero, Madrid', 'Malasaña-Universidad, Madrid', 'Marroquina, Madrid',
    'Media Legua, Madrid',
    'Mirasierra, Madrid', 'Moncloa, Madrid', 'Montecarmelo, Madrid', 'Moratalaz, Madrid', 'Moscardó, Madrid',
    'Niño Jesús, Madrid', 'Nueva España, Madrid', 'Nuevos Ministerios-Ríos Rosas, Madrid', 'Numancia, Madrid',
    'Opañel, Madrid', 'Orcasitas, Madrid', 'Pacífico, Madrid', 'Palacio, Madrid', 'Palomas, Madrid',
    'Palomeras Bajas, Madrid', 'Palomeras sureste, Madrid', 'Palos de Moguer, Madrid', 'Pau de Carabanchel, Madrid',
    'Pavones, Madrid', 'Peñagrande, Madrid', 'Pilar, Madrid', 'Pinar del Rey, Madrid', 'Portazgo, Madrid',
    'Pradolongo, Madrid', 'Prosperidad, Madrid', 'Pueblo Nuevo, Madrid', 'Puente de Vallecas, Madrid',
    'Puerta Bonita, Madrid', 'Puerta del Ángel, Madrid', 'Quintana, Madrid', 'Recoletos, Madrid', 'Retiro, Madrid',
    'San Andrés, Madrid', 'San Cristóbal, Madrid', 'San Diego, Madrid', 'San Fermín, Madrid', 'San Isidro, Madrid',
    'San Juan Bautista, Madrid', 'San Pascual, Madrid', 'Sanchinarro, Madrid', 'Santa Eugenia, Madrid', 'Sol, Madrid',
    'Tetuán, Madrid', 'Timón, Madrid', 'Trafalgar, Madrid', 'Tres Olivos - Valverde, Madrid', 'Usera, Madrid',
    'Valdeacederas, Madrid', 'Valdebebas - Valdefuentes, Madrid', 'Valdebernardo - Valderribas, Madrid',
    'Valdemarín, Madrid', 'Valdezarza, Madrid', 'Vallehermoso, Madrid', 'Ventas, Madrid', 'Ventilla-Almenara, Madrid',
    'Vicálvaro, Madrid', 'Villa de Vallecas, Madrid', 'Villaverde, Madrid', 'Vinateros, Madrid',
    'Virgen del Cortijo - Manoteras, Madrid', 'Vista Alegre, Madrid', 'Zofío, Madrid', 'Águilas, Madrid'
]


def load_model():
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

with open('scaler.pkl', 'rb') as model_file:
    scaler_loaded = pickle.load(model_file)

best_rf = load_model()

with open('scaler.pkl', 'rb') as model_file:
    scaler_loaded = pickle.load(model_file)

# Define the function for predicting price
def predict_price(location, sq_mt_built, built_year, has_parking):
    # Create a dictionary to hold the input data
    input_data = {}
    
    # Set the location dummy variable
    for loc in location_names:
        input_data[loc] = 1 if loc == location else 0

    # Add the other input features to the dictionary
    input_data['sq_mt_built'] = sq_mt_built
    input_data['built_year'] = built_year
    input_data['has_parking'] = has_parking

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Standardize the input data using the previously trained scaler
    input_scaled = scaler_loaded.transform(input_df)

    # Make the prediction using the model
    prediction = np.exp(best_rf.predict(input_scaled))[0]

    return prediction

# Streamlit app layout
st.title('Madrid Real Estate Price Predictor')

# User input fields
location = st.selectbox('Location', location_names)
# Get the other input values from the user
sq_mt_built = st.number_input('Square Meters Built', min_value=200)
built_year = st.number_input('Year Built', min_value = 1900)
has_parking = 1 if st.radio('Has Parking', ['Yes', 'No']) == 'Yes' else 0

# Make the prediction
prediction = predict_price(location, sq_mt_built, built_year, has_parking)
st.success(f'Predicted Price: {prediction:.2f} Euros')
