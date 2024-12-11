import streamlit as st #streamlit run server.py --server.port 8080
import pandas as pd
import pickle

# Loading the trained model, scaler, and label encoder
@st.cache_resource
def load_model():
    try:
        lableEncoder = pickle.load(open('app/lableEncoder.pkl', 'rb'))
        scaler = pickle.load(open('app/robust_scale.pkl', 'rb'))
        model = pickle.load(open('app/rf_classifier.pkl', 'rb'))
        return lableEncoder, scaler, model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None

lableEncoder, scaler, model = load_model()

def data_processing_and_output(dataframe):
    # Preprocessing the data
    columns_to_one_hot_encode = ['Passed','TestPlan','cosmetic_grade']

    # Loop through each column and apply One-Hot Encoding
    for column in columns_to_one_hot_encode:
        dataframe = dataframe.join(dataframe[column].str.get_dummies(sep=','))
        dataframe.drop(column, axis=1, inplace=True)

    # Select the columns you want to scale
    columns_to_scale = ['battery_cycle_count', 'battery_temperature', 'battery_total_operating_time']
    dataframe[columns_to_scale] = scaler.transform(dataframe[columns_to_scale])

    # Get the column names that the random forest was trained on
    training_data_columns = model.feature_names_in_

    # Reindex dataframe to match the training data columns
    dataframe = dataframe.reindex(columns=training_data_columns, fill_value=0)

    # Make predictions
    predictions = model.predict(dataframe)

    # Convert predictions to strings
    predictions = lableEncoder.inverse_transform(predictions)

    # Ensure predictions are returned as a list for compatibility with JSON
    return predictions.tolist()

# Check if model is loaded
if lableEncoder is not None and scaler is not None and model is not None:
    
    st.title("Android Grade Prediction")
    st.markdown("**Fill in the details below to predict the grade of the Android device:**")

    # Inputs for integer values
    st.subheader("Battery Information")
    battery_cycle_count = st.number_input('Battery Cycle Count', min_value=0, step=1)
    battery_temperature = st.number_input('Battery Temperature', min_value=0, step=1)
    battery_total_operating_time = st.number_input('Battery Total Operating Time', min_value=0, step=1)
    
    # Inputs for categorical values in 3 columns
    st.subheader("Device Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        TestPlan = st.radio('Test Plan', ['wholesale', 'wholesale_v2', 'insurance', 'ecomm_v2'])
    with col2:
        cosmetic_grade = st.radio('Cosmetic Grade', ['S4', 'S1', 'S2', 'S3', 'A', 'B', 'C', 'D'])
    with col3:
        b_and_p_qual = int(st.radio('Build and Performance Quality', ['0', '1']))

    # Input for checkboxes (multi-select) with heading
    st.subheader('Passed Components')
    Passed_options = [
        'VIBRATOR', 'LOUD_SPEAKER', 'EAR_SPEAKER', 'ALL_MICS', 'HEADSET_LR', 'HEADSET_MIC', 'HEADSET_INSERT', 'ACCELEROMETER', 'GYROSCOPE', 
        'MAGNETOMETER', 'LIGHT', 'FRONT_CAMERA', 'REAR_CAMERA', 'VIBRATE_SWITCH', 'KEY', 'WIRELESSCHARGING', 'FLASH', 'WIREDCHARGING', 
        'TOUCH', 'MULTITOUCH', 'PROXIMITY', 'BLUETOOTH', 'TRUE_DEPTH_CAMERA', 'DISPLAY_LINES', 'DISPLAY_BURN_IN', 'DISPLAY_HOT_PIXEL', 
        'DISPLAY_DARK_PIXEL', 'DISPLAY_LIGHT_LEAK', 'DISPLAY_WHITE_SPOTS', 'DISPLAY_DARK_REGION', 'DISPLAY_OTHER', 'NFC', 'DISPLAY'
    ]
    
    # Display checkboxes in multiple columns for better layout
    columns = st.columns(3)
    Passed = [option for index, option in enumerate(Passed_options) if columns[index % 3].checkbox(option)]

    # Button to submit the data
    if st.button('Predict Grade'):
        # Create a DataFrame with the input data
        data_dict = {
            'Passed': ','.join(Passed),
            'TestPlan': TestPlan,
            'cosmetic_grade': cosmetic_grade,
            'b_and_p_qual': b_and_p_qual,
            'battery_cycle_count': battery_cycle_count,
            'battery_temperature': battery_temperature,
            'battery_total_operating_time': battery_total_operating_time
        }
        df = pd.DataFrame([data_dict])
        
        # Display the DataFrame
        st.subheader("Input Data")
        st.dataframe(df)
        
        # Process the data and make predictions
        try:
            output = data_processing_and_output(df)
            st.success(f"Predicted Grade: {output[0]}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
