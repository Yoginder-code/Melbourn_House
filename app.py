import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Load the trained model and other necessary components ---
try:
    # Load the entire pipeline, which includes the preprocessor and the model
    pipeline = joblib.load('ridge_model_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: The model file 'ridge_model_pipeline.joblib' was not found.")
    st.write("Please ensure the model is saved in the same directory as this app.py file.")
    st.stop() # Stop the app if the model is not found

# --- 2. Streamlit App Interface ---
st.set_page_config(page_title="Melbourne Housing Price Predictor", layout="wide", page_icon="🏡")

# --- Header Section ---
st.title("🏡 Melbourne Housing Price Predictor")
st.markdown("""
### Predict Property Prices in Melbourne's Eastern Suburbs
This application uses machine learning to estimate property values in Box Hill, Burwood/Burwood East, and Doncaster/Doncaster East.
""")

# --- Project Information Section ---
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    #### Project Overview
    This housing price prediction system was developed as part of a comprehensive data science project analyzing Melbourne's eastern suburbs real estate market.
    
    #### Methodology
    1. **Data Collection**: 150+ property records collected from realestate.com.au
    2. **Feature Engineering**: Created predictive features like room ratios and parking efficiency
    3. **Model Development**: Multiple regression models evaluated using cross-validation
    4. **Feature Importance**: SHAP analysis to identify key price drivers
    5. **Deployment**: Web application for interactive price predictions
    
    #### Suburbs Covered
    - **Box Hill**: Major transport hub with excellent amenities
    - **Burwood & Burwood East**: Education hub near Deakin University
    - **Doncaster & Doncaster East**: Family-friendly with premium schools
    
    #### Technical Stack
    - Python, Scikit-learn, Pandas, NumPy
    - Streamlit for web deployment
    - SHAP for model interpretability
    """)

# --- Visualization Section ---
st.header("📊 Market Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Prices by Suburb**")
    # Sample data - replace with your actual data
    suburb_prices = pd.DataFrame({
        'Suburb': ['Box Hill', 'Burwood Region', 'Doncaster Region'],
        'Average Price': [1250000, 1150000, 1350000]
    })
    fig1 = px.bar(suburb_prices, x='Suburb', y='Average Price', 
                 color='Suburb', title="Average Housing Prices by Suburb")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("**Price Distribution**")
    # Sample data - replace with your actual data
    np.random.seed(42)
    prices = np.random.normal(1200000, 300000, 1000)
    fig2 = px.histogram(x=prices, nbins=30, title="Overall Price Distribution",
                       labels={'x': 'Price', 'y': 'Frequency'})
    st.plotly_chart(fig2, use_container_width=True)

# --- Input Section ---
st.header("🔍 Property Details")
st.markdown("Enter your property features below to get a price prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic Information")
    property_type = st.selectbox(
        "Property Type",
        ('House', 'Apartment', 'Townhouse', 'Unit'),
        help="Select the type of property"
    )
    suburb = st.selectbox(
        "Suburb",
        ('Box Hill', 'Burwood & Burwood East', 'Doncaster & Doncaster East'),
        help="Select the suburb location"
    )

with col2:
    st.subheader("Room Configuration")
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3,
                        help="Select the number of bedrooms")
    bathrooms = st.slider("Number of Bathrooms", 1, 10, 2,
                         help="Select the number of bathrooms")
    total_rooms = st.slider("Total Number of Rooms", 1, 20, 5,
                           help="Bedrooms + bathrooms + other rooms")

with col3:
    st.subheader("Additional Features")
    parking_space = st.slider("Number of Parking Spaces", 0, 5, 1,
                             help="Number of dedicated parking spaces")
    # Additional features could be added here based on your dataset

# --- Prediction Section ---
st.header("💰 Price Prediction")

if st.button("Predict Price", type="primary", help="Click to calculate the estimated price"):
    # Calculate engineered features from user input
    bath_bed_ratio = bathrooms / bedrooms if bedrooms > 0 else 0
    parking_per_bedroom = parking_space / bedrooms if bedrooms > 0 else 0
    
    # Create a DataFrame for the new property
    new_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'Parking space': parking_space,
        'total_rooms': total_rooms,
        'property type': property_type,
        'Suburb': suburb,
        'bath_bed_ratio': bath_bed_ratio,
        'parking_per_bedroom': parking_per_bedroom
    }])
    
    # Predict the price using the loaded pipeline
    try:
        predicted_price = pipeline.predict(new_data)[0]
        
        # Display the result with styling
        st.success("### Prediction Result")
        
        # Create a metric display
        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric(label="Estimated Property Value", 
                     value=f"${predicted_price:,.2f}",
                     delta="Model Prediction")
        
        # Additional insights
        st.info("""
        **Note**: This prediction is based on machine learning models trained on historical data. 
        Actual market prices may vary based on property condition, exact location, and market conditions.
        """)
        
        # Show feature impact (simulated - replace with your actual SHAP values)
        st.subheader("📈 Feature Impact on Price")
        feature_impact = pd.DataFrame({
            'Feature': ['Bedrooms', 'Bathrooms', 'Parking', 'Property Type', 'Suburb'],
            'Impact': [0.32, 0.28, 0.15, 0.12, 0.13]
        })
        fig3 = px.bar(feature_impact, x='Impact', y='Feature', orientation='h',
                     title="Relative Feature Importance")
        st.plotly_chart(fig3, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer Section ---
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool provides estimates only and should not be considered as professional property advice. 
Always consult with real estate professionals for accurate valuations.
""")