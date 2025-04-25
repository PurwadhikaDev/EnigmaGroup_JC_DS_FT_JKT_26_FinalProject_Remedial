import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from utils import get_feature_importance, plot_feature_importance, get_feature_contributions

# Must be the first Streamlit command
st.set_page_config(
    page_title="Churn Prediction - IndoHome", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan konfigurasi
with open('deployment/final_feature_list.json') as f:
    final_features = json.load(f)

with open('deployment/threshold.json') as f:
    threshold = json.load(f)['threshold']

model = joblib.load('deployment/final_model_pipeline.pkl')

# Definisikan rate konversi USD ke IDR
USD_TO_IDR = 15700  # Rate konversi USD ke IDR

# Function untuk konversi dan format currency
def format_currency(amount, currency="Rp"):
    if currency == "Rp":
        return f"{currency} {amount:,.0f}"
    return f"{currency} {amount:.2f}"

# Function untuk konversi USD ke IDR
def usd_to_idr(usd_amount):
    return usd_amount * USD_TO_IDR

# Updated custom styling with modern sidebar design
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 0 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #f0f0f0;
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        padding: 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        background-color: transparent;
        padding: 0.5rem 0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        color: #2C3E50;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.5rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] > div > label[data-checked="true"] {
        background-color: #E60000;
        color: white;
    }
    
    /* Logo styling */
    .sidebar-logo {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .sidebar-logo img {
        max-width: 150px;
        margin: 0 auto;
        display: block;
    }
    
    /* Custom header styling */
    .custom-header {
        background: white;
        border-left: 4px solid #E60000;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 8px;
    }
    
    .custom-header h1 {
        color: #2C3E50;
        font-family: 'Segoe UI', sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }
    
    .custom-header p {
        color: #7F8C8D;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        margin-top: 0.5rem;
    }
    
    /* Section header styling */
    .section-header {
        background: white;
        border-left: 4px solid #E60000;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .section-header h2 {
        color: #2C3E50;
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: 600;
        margin: 0;
    }
    
    /* Metric container styling */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 1.5rem;
        background: white;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #E60000;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 4px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #A93226;
    }
    
    /* Tooltip styling */
    .tooltip-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: help;
    }
    
    /* Step indicator styling */
    .step-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    
    .step-number {
        background: #E60000;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
    }
    
    /* Success/Error message styling */
    .message-container {
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #E8F5E9;
        border: 1px solid #81C784;
        color: #2E7D32;
    }
    
    .error-message {
        background: #FFEBEE;
        border: 1px solid #E57373;
        color: #C62828;
    }
    
    /* Input field styling */
    .stNumberInput {
        width: 100%;
    }
    
    .stNumberInput > div > div > input {
        padding-right: 40px !important;  /* Make room for the increment buttons */
    }
    
    /* Hide the "Press Enter to submit form" text */
    .stNumberInput div[data-baseweb="input"] div[data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    
    /* Clean up form layout */
    [data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Metric container refinements */
    .metric-container {
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Remove unnecessary margins and padding */
    .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Clean up success/error messages */
    .message-container {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 6px;
        font-size: 14px;
    }
    
    /* Remove emoji from headers */
    .custom-header h1,
    .section-header h2 {
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Form field styling */
    .form-field {
        margin-bottom: 1.5rem;
    }
    
    .field-description {
        font-size: 12px;
        color: #666;
        margin-top: 2px;
        margin-bottom: 8px;
    }
    
    /* Hide empty metric containers */
    .metric-container:empty {
        display: none;
    }
    
    /* Input field container */
    .input-container {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #eee;
        margin-bottom: 1rem;
    }
    
    /* Dashboard metric boxes */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-box h3 {
        color: #2C3E50;
        font-size: 14px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #E60000;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
    }
    
    /* Segment boxes */
    .segment-box {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .segment-box h4 {
        margin: 0;
        font-size: 16px;
        color: #2C3E50;
    }
    
    .segment-count {
        font-size: 24px;
        font-weight: 600;
        margin: 8px 0;
    }
    
    .segment-percent {
        font-size: 14px;
        color: #666;
        margin: 0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Updated sidebar with modern design
st.sidebar.markdown("""
<div class="sidebar-logo">
    <span style="font-size: 24px; font-weight: 700; color: #2C3E50;">
        <span style="color: #E60000;">Indo</span>Home
    </span>
    <div style="font-size: 12px; color: #7F8C8D; margin-top: 4px;">
        Churn Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

# Updated sidebar navigation
st.sidebar.markdown('<div style="margin-bottom: 1rem; color: #7F8C8D; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "Customer Segmentation", "Predict", "Batch Prediction", "Model Interpretation", "About"],
    label_visibility="collapsed"
)

# Version and author info at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="position: fixed; bottom: 0; padding: 1rem; font-size: 12px; color: #7F8C8D;">
    <div style="margin-bottom: 0.5rem;">Version 1.0</div>
    <div>Made by Rina Adibah</div>
    <a href="https://www.linkedin.com/in/rina-adibah/" style="color: #E60000; text-decoration: none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

# Custom function for page headers
def create_page_header(title, description=""):
    st.markdown(f"""
    <div class="custom-header">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

# Custom function for section headers
def create_section_header(title):
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)

# Custom Header HTML
st.markdown("""
<div style="width: 100%; background: linear-gradient(to right, #E60000, #A93226); border-left: 12px solid #B03A2E; border-radius: 12px; padding: 30px 25px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); text-align: center;">
  <div style="font-size: 40px; font-family: 'Segoe UI', sans-serif; font-weight: 800; margin-bottom: 10px; letter-spacing: -1px;">
    <span style="color: #FADBD8;">📡</span>
    <span style="color: #ffffff;">Indo</span><span style="color: #D5D8DC;">Home</span>
  </div>

  <h1 style="font-size: 30px; color: #FDFEFE; font-weight: 600; font-family: 'Segoe UI', sans-serif; margin: 10px 0 5px;">
    Telco Customer Churn Prediction
  </h1>

  <p style="font-size: 16px; color: #F2F4F4; font-weight: 400; font-family: 'Segoe UI', sans-serif; margin-top: 0;">
    Empowering Customer Retention Strategy in Indonesia's Telecommunication Industry through Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

# ===================== HOME =====================
if page == "Home":
    create_page_header(
        "IndoHome Customer Churn Prediction System",
        "Customer Churn Prediction System for Marketing and Management Teams"
    )
    
    with st.expander("📖 Application Guide", expanded=True):
        st.markdown("""
        <div class="step-container">
            <div><span class="step-number">1</span> <strong>Individual Prediction</strong></div>
            <p>Use the "Predict" menu to predict the churn probability for a single customer:</p>
            <ul>
                <li>Enter customer data in the provided form</li>
                <li>System will display risk level and recommended actions</li>
                <li>Analyze prediction factors in detail using SHAP graph</li>
            </ul>
        </div>
        
        <div class="step-container">
            <span class="step-number">2</span> <strong>Batch Prediction</strong>
            <p>Use the "Batch Prediction" menu for multiple customer analysis:</p>
            <ul>
                <li>Upload CSV file with the required format</li>
                <li>Download prediction results in CSV format</li>
                <li>Analyze trends and patterns from batch predictions</li>
            </ul>
        </div>
        
        <div class="step-container">
            <span class="step-number">3</span> <strong>Model Interpretation</strong>
            <p>Understand how the model works through the "Model Interpretation" menu:</p>
            <ul>
                <li>View the most influential factors affecting churn</li>
                <li>Use these insights for strategic decision making</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            create_section_header("Overview")
            st.write("""
                This system is designed to help IndoHome's Marketing and Management teams 
                identify customers with high churn risk through predictive analytics.
            """)
            
        with col2:
            create_section_header("Key Features")
            st.write("""
                - Individual customer churn prediction
                - Batch prediction via CSV upload
                - Risk-based action recommendations
            """)

# ===================== PREDICT =====================
elif page == "Predict":
    create_page_header(
        "Customer Churn Prediction",
        "Customer Data Input Form for Churn Risk Analysis"
    )

    with st.form(key="churn_form"):
        create_section_header("Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<p class="field-description">Basic Customer Information</p>', unsafe_allow_html=True)
            
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                help="Duration of subscription in months"
            )
            
            monthly_charges_idr = st.number_input(
                "Monthly Charges (Rp)",
                min_value=0,
                max_value=10000000,
                value=1000000,
                step=100000,
                format="%d",
                help="Monthly charges in Rupiah"
            )
            
            total_charges_idr = st.number_input(
                "Total Charges (Rp)",
                min_value=0,
                max_value=100000000,
                value=12000000,
                step=1000000,
                format="%d",
                help="Total charges since subscription start in Rupiah"
            )
            
            # Convert back to USD for model
            monthly_charges = monthly_charges_idr / USD_TO_IDR
            total_charges = total_charges_idr / USD_TO_IDR
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<p class="field-description">Customer Profile</p>', unsafe_allow_html=True)
            
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                help="Customer's gender"
            )
            
            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                help="Age > 65 years"
            )
            
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"],
                help="Marital status"
            )
            
            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
                help="Has dependents"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<p class="field-description">Services Used</p>', unsafe_allow_html=True)
            
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                help="Type of internet service"
            )
            
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                help="Subscription contract type"
            )
            
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                help="Payment method used"
            )
            
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"],
                help="Digital billing usage"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Additional Services in an expander
        with st.expander("Additional Services", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                phone_service = st.selectbox(
                    "Phone Service",
                    ["No", "Yes"],
                    help="Phone service usage"
                )
                
                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    ["No", "Yes", "No phone service"],
                    help="Multiple phone lines usage"
                )
                
                online_security = st.selectbox(
                    "Online Security",
                    ["No", "Yes", "No internet service"],
                    help="Online security service"
                )
                
                online_backup = st.selectbox(
                    "Online Backup",
                    ["No", "Yes", "No internet service"],
                    help="Online backup service"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                device_protection = st.selectbox(
                    "Device Protection",
                    ["No", "Yes", "No internet service"],
                    help="Device protection service"
                )
                
                tech_support = st.selectbox(
                    "Tech Support",
                    ["No", "Yes", "No internet service"],
                    help="Technical support service"
                )
                
                streaming_tv = st.selectbox(
                    "TV Streaming",
                    ["No", "Yes", "No internet service"],
                    help="TV streaming service"
                )
                
                streaming_movies = st.selectbox(
                    "Movie Streaming",
                    ["No", "Yes", "No internet service"],
                    help="Movie streaming service"
                )
                st.markdown('</div>', unsafe_allow_html=True)

        submit = st.form_submit_button("Predict Churn")

    if submit:
        with st.spinner('Processing prediction...'):
            input_data = {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "gender": gender,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_service,
                "InternetService": internet_service,
                "Contract": contract,
                "PaymentMethod": payment_method,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "TotalServices": [online_security, online_backup, device_protection,
                                  tech_support, streaming_tv, streaming_movies].count("Yes"),
                "TenureGroup": "Baru" if tenure <= 12 else "Menengah" if tenure <= 24 else "Lama",
                "MultipleLines": multiple_lines,
                "PaperlessBilling": paperless_billing
            }

            input_df = pd.DataFrame([input_data])
            prob = model.predict_proba(input_df)[0][1]
            label = int(prob >= threshold)

            create_section_header("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                # Risk gauge visualization
                risk_color = "green"
                if prob > 0.8:
                    risk_color = "red"
                elif prob > 0.6:
                    risk_color = "orange"
                elif prob >= 0.43:
                    risk_color = "gold"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk Level", 'font': {'size': 20, 'family': 'Segoe UI'}},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 0.43], 'color': 'lightgreen'},
                            {'range': [0.43, 0.6], 'color': 'khaki'},
                            {'range': [0.6, 0.8], 'color': 'orange'},
                            {'range': [0.8, 1.0], 'color': 'tomato'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.43
                        }
                    }
                ))
                st.plotly_chart(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                create_section_header("Recommended Actions")
                st.metric(label="Probability of Churn", value=f"{prob:.2%}")

                if prob > 0.8:
                    st.error("❗ Customer predicted to be at HIGH RISK of churning. **Recommendation:** Send special offers or retention discounts.")
                elif prob > 0.6:
                    st.warning("⚠️ Customer in MEDIUM RISK zone. **Recommendation:** Offer loyalty program or service bundling.")
                elif prob >= 0.43:
                    st.info("ℹ️ Customer in CAUTION zone. **Recommendation:** Monitor and perform light intervention if needed.")
                else:
                    st.success("✅ Customer predicted to be LOYAL. No intervention needed at this time.")
                st.markdown('</div>', unsafe_allow_html=True)

        # Add feature contribution analysis
        st.subheader("Feature Contributions to Prediction")
        
        # Get feature names after preprocessing
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        # Calculate individual feature contributions
        contributions_df = get_feature_contributions(input_df, model, feature_names)
        
        # Display top 10 contributions
        st.markdown("Top 10 Most Influential Features for This Prediction:")
        
        # Style the dataframe
        def color_contributions(val):
            if pd.isna(val):  # Handle NaN values
                return ''
            color = '#B2182B' if val > 0 else '#2166AC'
            return f'color: {color}'
        
        # Format and display the dataframe
        display_df = contributions_df.head(10).copy()
        
        # Add percentage contribution column
        total_abs_contrib = display_df['Contribution'].abs().sum()
        display_df['Impact'] = (display_df['Contribution'].abs() / total_abs_contrib * 100).round(1).astype(str) + '%'
        
        # Reorder columns for display
        display_df = display_df[['Feature', 'Value', 'Coefficient', 'Contribution', 'Impact']]
        
        # Style the dataframe
        styled_df = display_df.style\
            .applymap(color_contributions, subset=['Contribution'])\
            .format({
                'Value': '{:.4f}',
                'Coefficient': '{:.4f}',
                'Contribution': '{:.4f}'
            })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Understanding the Results:**
        - **Feature**: The input variable
        - **Value**: The standardized/encoded input value
        - **Coefficient**: The feature's weight in the model
        - **Contribution**: Value × Coefficient (positive values increase churn probability)
        - **Impact**: Percentage of total contribution magnitude
        """)

# ===================== BATCH PREDICTION =====================
elif page == "Batch Prediction":
    create_page_header(
        "Batch Churn Prediction",
        "Upload CSV File for Multiple Customer Predictions"
    )
    
    st.info("""
    ℹ️ **Required CSV File Format:**
    - File must contain columns: tenure, MonthlyCharges, TotalCharges, etc.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            with st.spinner('Processing file...'):
                df = pd.read_csv(uploaded_file)

                df["TotalServices"] = (df[[
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"
                ]] == "Yes").sum(axis=1)

                df["TenureGroup"] = df["tenure"].apply(
                    lambda x: "Baru" if x <= 12 else "Menengah" if x <= 24 else "Lama"
                )

                probs = model.predict_proba(df)[:, 1]
                labels = (probs >= threshold).astype(int)

                df["churn_probability"] = probs
                df["predicted_label"] = labels
                df["status"] = df["predicted_label"].map({0: "Not Churning", 1: "CHURNING"})

                st.success(f"Successfully predicted for {len(df)} customers.")
                st.dataframe(df[["churn_probability", "status"] + list(df.columns[:3])])

                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button("Unduh Hasil Prediksi", csv_out, "hasil_churn_batch.csv", "text/csv")

                st.markdown("""
                <div class="message-container success-message">
                    ✅ File processed successfully! Please download the prediction results.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown("""
            <div class="message-container error-message">
                Error processing file. Please check the file format.
            </div>
            """, unsafe_allow_html=True)
            st.exception(e)

# ===================== INTERPRETASI MODEL =====================
elif page == "Model Interpretation":
    create_page_header(
        "Model Interpretation",
        "Understanding Feature Importance in the Logistic Regression Model"
    )

    try:
        # Load model and get feature names
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        # Get global feature importance
        importance_df = get_feature_importance(model, feature_names)
        
        if importance_df.empty:
            st.error("Unable to calculate feature importance. Please check the model.")
            st.stop()
        
        # Display feature importance table
        st.subheader("Global Feature Importance")
        st.markdown("""
        This table shows how each feature influences customer churn:
        - **Coefficient**: Log-odds impact on churn probability (negative = reduces churn, positive = increases churn)
        - **Odds Ratio**: How much the odds of churning change when the feature increases by one unit
        """)
        
        # Style the dataframe
        def color_coefficients(val):
            try:
                val = float(val)
                color = '#B2182B' if val > 0 else '#2166AC'
                return f'color: {color}'
            except:
                return ''
        
        styled_df = importance_df.style\
            .applymap(color_coefficients, subset=['Coefficient'])\
            .format({
                'Coefficient': '{:.4f}',
                'Odds_Ratio': '{:.4f}'
            })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Plot feature importance
        st.subheader("Top 10 Most Influential Features")
        st.markdown("""
        This plot shows the top 10 features with the strongest influence on churn:
        - **Red bars**: Features that increase churn probability
        - **Blue bars**: Features that decrease churn probability
        - **Bar length**: Magnitude of feature impact
        """)
        
        fig = plot_feature_importance(importance_df)
        if fig is not None:
            st.pyplot(fig)
        
        # Add interpretation summary
        st.subheader("Key Insights")
        
        # Get top positive and negative features
        pos_features = importance_df[importance_df['Coefficient'] > 0]
        neg_features = importance_df[importance_df['Coefficient'] < 0]
        
        if not pos_features.empty:
            st.markdown("**Top Factors Increasing Churn Risk:**")
            for _, row in pos_features.head(3).iterrows():
                st.markdown(f"- **{row['Feature']}**: Increases odds of churning by {(row['Odds_Ratio']-1)*100:.1f}%")
        
        if not neg_features.empty:
            st.markdown("**Top Factors Decreasing Churn Risk:**")
            for _, row in neg_features.head(3).iterrows():
                st.markdown(f"- **{row['Feature']}**: Decreases odds of churning by {(1-row['Odds_Ratio'])*100:.1f}%")

    except Exception as e:
        st.error("Error loading model interpretation data.")
        st.exception(e)

# ===================== DASHBOARD =====================
elif page == "Dashboard":
    create_page_header(
        "Dashboard Analytics",
        "Ringkasan Metrik dan Analisis Churn Pelanggan"
    )
    
    try:
        df = pd.read_csv("deployment/telco_customer_data.csv")
        
        # Konversi MonthlyCharges dan TotalCharges ke IDR
        df['MonthlyCharges_IDR'] = df['MonthlyCharges'] * USD_TO_IDR
        df['TotalCharges_IDR'] = df['TotalCharges'].astype(float) * USD_TO_IDR
        
        # Calculate key metrics
        total_customers = len(df)
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        avg_monthly_idr = df['MonthlyCharges_IDR'].mean()
        avg_tenure = df['tenure'].mean()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3>Total Customers</h3>
                <p class="metric-value">{:,}</p>
            </div>
            """.format(total_customers), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3>Churn Rate</h3>
                <p class="metric-value">{:.1f}%</p>
            </div>
            """.format(churn_rate), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3>Average Monthly Charges</h3>
                <p class="metric-value">{}</p>
            </div>
            """.format(format_currency(avg_monthly_idr)), unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-box">
                <h3>Average Tenure</h3>
                <p class="metric-value">{:.1f} months</p>
            </div>
            """.format(avg_tenure), unsafe_allow_html=True)
        
        # Churn Analysis by Contract Type
        st.subheader("Churn Analysis by Contract Type")
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        
        fig = go.Figure()
        fig.add_bar(
            name="Churn",
            x=contract_churn.index,
            y=contract_churn['Yes'],
            marker_color='#E60000'
        )
        fig.add_bar(
            name="Retained",
            x=contract_churn.index,
            y=contract_churn['No'],
            marker_color='#28a745'
        )
        
        fig.update_layout(
            barmode='stack',
            title="Churn Rate by Contract Type",
            xaxis_title="Contract Type",
            yaxis_title="Percentage (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Charges Distribution (in IDR)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Charges Distribution")
            fig = go.Figure()
            fig.add_histogram(
                x=df['MonthlyCharges_IDR'],
                nbinsx=30,
                marker_color='#E60000'
            )
            fig.update_layout(
                title="Distribution of Monthly Charges",
                xaxis_title="Monthly Charges (Rp)",
                yaxis_title="Count",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn by Payment Method")
            payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
            fig = go.Figure(data=[go.Pie(
                labels=payment_churn.index,
                values=payment_churn['Yes'],
                hole=.3
            )])
            fig.update_layout(
                title="Churn Distribution by Payment Method"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error("Error loading dashboard data. Please check the data source.")
        st.exception(e)

# ===================== CUSTOMER SEGMENTATION =====================
elif page == "Customer Segmentation":
    create_page_header(
        "Customer Segmentation",
        "Segmentasi Pelanggan Berdasarkan Risiko Churn"
    )
    
    try:
        # Load data
        df = pd.read_csv("deployment/telco_customer_data.csv")
        
        # Konversi ke IDR
        df['MonthlyCharges_IDR'] = df['MonthlyCharges'] * USD_TO_IDR
        df['TotalCharges_IDR'] = df['TotalCharges'].astype(float) * USD_TO_IDR
        
        # Calculate churn probability using the model
        df["TotalServices"] = (df[[
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]] == "Yes").sum(axis=1)

        df["TenureGroup"] = df["tenure"].apply(
            lambda x: "Baru" if x <= 12 else "Menengah" if x <= 24 else "Lama"
        )
        
        # Calculate churn probabilities
        churn_probabilities = model.predict_proba(df)[:, 1]
        df['churn_probability'] = churn_probabilities
        
        # Define risk segments
        def get_risk_segment(prob):
            if prob >= 0.8:
                return "High Risk"
            elif prob >= 0.6:
                return "Medium Risk"
            elif prob >= 0.43:
                return "Low Risk"
            return "Loyal"
        
        # Add risk segment column
        df['risk_segment'] = df['churn_probability'].apply(get_risk_segment)
        
        # Display segment summary
        segments = df['risk_segment'].value_counts()
        
        # Segment metrics
        st.subheader("Customer Risk Segments")
        
        cols = st.columns(len(segments))
        colors = {'High Risk': '#dc3545', 'Medium Risk': '#ffc107', 
                 'Low Risk': '#17a2b8', 'Loyal': '#28a745'}
        
        for col, (segment, count) in zip(cols, segments.items()):
            with col:
                st.markdown(f"""
                <div class="segment-box" style="background-color: {colors[segment]}20; border-left: 4px solid {colors[segment]}">
                    <h4>{segment}</h4>
                    <p class="segment-count">{count:,}</p>
                    <p class="segment-percent">{count/len(df)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Segment characteristics
        st.subheader("Segment Characteristics")
        segment_stats = df.groupby('risk_segment').agg({
            'tenure': 'mean',
            'MonthlyCharges_IDR': 'mean',
            'TotalCharges_IDR': 'mean',
            'Contract': lambda x: x.value_counts().index[0],
            'InternetService': lambda x: x.value_counts().index[0],
            'churn_probability': 'mean'
        }).round(2)
        
        # Format currency columns
        segment_stats['MonthlyCharges_IDR'] = segment_stats['MonthlyCharges_IDR'].apply(lambda x: format_currency(x))
        segment_stats['TotalCharges_IDR'] = segment_stats['TotalCharges_IDR'].apply(lambda x: format_currency(x))
        
        st.dataframe(
            segment_stats,
            column_config={
                "tenure": "Avg. Tenure (months)",
                "MonthlyCharges_IDR": "Avg. Monthly Charges",
                "TotalCharges_IDR": "Avg. Total Charges",
                "Contract": "Most Common Contract",
                "InternetService": "Most Common Internet Service",
                "churn_probability": "Avg. Churn Probability"
            }
        )
        
        # Segment Analysis Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Contract Type by Risk Segment")
            contract_segment = pd.crosstab(df['risk_segment'], df['Contract'])
            fig = go.Figure(data=[
                go.Bar(name=contract, x=contract_segment.index, y=contract_segment[contract])
                for contract in contract_segment.columns
            ])
            fig.update_layout(
                barmode='stack',
                title="Distribution of Contract Types Across Risk Segments",
                xaxis_title="Risk Segment",
                yaxis_title="Number of Customers",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Internet Service by Risk Segment")
            internet_segment = pd.crosstab(df['risk_segment'], df['InternetService'])
            fig = go.Figure(data=[
                go.Bar(name=service, x=internet_segment.index, y=internet_segment[service])
                for service in internet_segment.columns
            ])
            fig.update_layout(
                barmode='stack',
                title="Distribution of Internet Services Across Risk Segments",
                xaxis_title="Risk Segment",
                yaxis_title="Number of Customers",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Additional insights
        st.subheader("Key Insights")
        
        # Update box plot for monthly charges
        fig = go.Figure()
        for segment in df['risk_segment'].unique():
            segment_data = df[df['risk_segment'] == segment]
            fig.add_box(
                y=segment_data['MonthlyCharges_IDR'],
                name=segment,
                boxpoints='outliers',
                marker_color=colors[segment]
            )
        
        fig.update_layout(
            title="Monthly Charges Distribution by Risk Segment",
            yaxis_title="Monthly Charges (Rp)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error("Error loading segmentation data. Please check the data source.")
        st.exception(e)

# ===================== ABOUT =====================
elif page == "About":
    st.title("About the Application")
    st.markdown("""
    This application is built to support IndoHome's Marketing and CMO teams in detecting 
    potential churning customers using a data-driven predictive approach.

    **Model:** Logistic Regression + ADASYN  
    **Optimal Threshold:** 0.43 (optimized with F2-Score)  
    **Deployment:** Streamlit WebApp  
    """)

    st.markdown("---")
    st.subheader("Contact")
    st.markdown("""
    **Name:** Rina Adibah
                
    **LinkedIn:** [Rina Adibah](https://www.linkedin.com/in/rina-adibah/)
                  
    **Email:** rina.adibah.011@gmail.com
    """)

    st.markdown("""
    This application was developed as part of the Data Science Track final project  
    and aims to demonstrate the application of Machine Learning for real business problems.  
    """)

