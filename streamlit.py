import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# Page configuration
st.set_page_config(layout="wide")

# Load your pre-trained model
with open('best_xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# Load the feature importance DataFrame
importance_df = load_feature_importance("feature_importance.xlsx")  # Replace with your file path

def get_user_input():
    st.sidebar.header("Property details")
    propertyType = st.sidebar.selectbox("Property Type", ["house", "townhouse", "other"])
    suburb = st.sidebar.text_input("Suburb", value="")
    postCode = st.sidebar.text_input("Postcode", value="")
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, value=3, step=1)
    bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, value=1, step=1)
    parkingSpaces = st.sidebar.number_input("Parking spaces", min_value=0, value=1, step=1)
    landSize = st.sidebar.number_input("Land size (sqm)", min_value=0.0, value=200.0, step=1.0, format="%.1f")
    indoorFeatures = st.sidebar.number_input("Indoor features", min_value=0, value=0, step=1)
    outdoorFeatures = st.sidebar.number_input("Outdoor features", min_value=0, value=0, step=1)
    latitude = st.sidebar.text_input("Latitude", value="")
    longitude = st.sidebar.text_input("Longitude", value="")

    user_data = {
        "propertyType": propertyType,
        "suburb": suburb.strip(),
        "postCode": postCode.strip(),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "parkingSpaces": int(parkingSpaces),
        "landSize": float(landSize),
        "indoorFeatures": int(indoorFeatures),
        "outdoorFeatures": int(outdoorFeatures),
        "totalFeatures": int(indoorFeatures) + int(outdoorFeatures),
        "latitude": float(latitude) if str(latitude).strip() != "" else np.nan,
        "longitude": float(longitude) if str(longitude).strip() != "" else np.nan,
    }
    return user_data

# Sidebar setup
st.sidebar.header('Housing Features')

# Centered title
st.markdown("<h1 style='text-align: center;'>Melbourne Housing Price Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")

    # Sort feature importance DataFrame by 'Importance Score'
    importance_sorted = importance_df.sort_values(by='Importance Score', ascending=True)
    
    # Create interactive bar chart with Plotly
    fig = px.bar(
        importance_sorted,
        x='Importance Score',
        y='Features',
        orientation='h',
        title="Feature Importance",
        labels={'Importance Score': 'Importance', 'Features': 'Feature'},
        text='Importance Score',
        color_discrete_sequence=["#156371"]  # Custom bar color
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Features",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Right column: Prediction Interface
with right_col:
    st.header("Predict Housing Price")
    
    # User inputs from sidebar
    user_data = get_user_input()

    def prepare_input(data, feature_list):
        # accept either a dict-like user_data or a single-row DataFrame
        if isinstance(data, pd.DataFrame):
            row = data.iloc[0].to_dict()
        else:
            row = dict(data or {})

        # small haversine helper (km)
        def haversine_km(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            return R * 2 * np.arcsin(np.sqrt(a))

        # CBD coords (same as used in notebook)
        cbd_lat, cbd_lon = -37.810272, 144.962646

        # read basic values with safe defaults
        prop = str(row.get("propertyType", "")).strip().lower()
        pc = str(row.get("postCode", "")).strip()
        suburb = str(row.get("suburb", "")).strip()

        def to_float(v):
            try:
                return float(v)
            except Exception:
                return np.nan

        def to_int(v, default=0):
            try:
                return int(v)
            except Exception:
                return default

        fr = {}
        fr["landSize"] = to_float(row.get("landSize", np.nan))
        fr["latitude"] = to_float(row.get("latitude", np.nan))
        fr["longitude"] = to_float(row.get("longitude", np.nan))
        fr["bedrooms"] = to_int(row.get("bedrooms", 0), 0)
        fr["bathrooms"] = to_int(row.get("bathrooms", 0), 0)
        fr["parkingSpaces"] = to_int(row.get("parkingSpaces", 0), 0)
        fr["outdoorFeatures"] = to_int(row.get("outdoorFeatures", 0), 0)
        fr["indoorFeatures"] = to_int(row.get("indoorFeatures", 0), 0)
        # totalFeatures prefer provided, else sum indoor/outdoor
        if row.get("totalFeatures") is not None:
            fr["totalFeatures"] = to_int(row.get("totalFeatures"), fr["indoorFeatures"] + fr["outdoorFeatures"])
        else:
            fr["totalFeatures"] = fr["indoorFeatures"] + fr["outdoorFeatures"]

        # one-hot property types (keep same three used in training features)
        fr["propertyType_house"] = 1 if prop == "house" else 0
        fr["propertyType_other"] = 1 if prop == "other" else 0
        fr["propertyType_townhouse"] = 1 if prop == "townhouse" else 0

        # postcode one-hot keys used in training
        postcode_keys = ["3103", "3124", "3125", "3127", "3128"]
        for p in postcode_keys:
            fr[f"postCode_{p}"] = 1 if pc == p else 0

        # suburb one-hot keys used in training (match case-insensitive)
        suburb_keys = ["Balwyn", "Box Hill", "Burwood", "Camberwell", "Surrey Hills"]
        for s in suburb_keys:
            fr[f"suburb_{s}"] = 1 if suburb.lower() == s.lower() else 0

        # date-derived features
        ds = row.get("dateSold", None)
        if ds is None or (isinstance(ds, float) and np.isnan(ds)):
            ds_ts = pd.Timestamp.today()
        else:
            ds_ts = pd.to_datetime(ds, errors="coerce")
            if pd.isna(ds_ts):
                ds_ts = pd.Timestamp.today()
        fr["soldYear"] = int(ds_ts.year)
        fr["soldMonth"] = int(ds_ts.month)
        fr["soldWeekday"] = int(ds_ts.weekday())

        # distances
        if not pd.isna(fr["latitude"]) and not pd.isna(fr["longitude"]):
            fr["distanceToCBD"] = haversine_km(fr["latitude"], fr["longitude"], cbd_lat, cbd_lon)
        else:
            fr["distanceToCBD"] = np.nan
        # distanceToNearest requires dataset coordinates -> leave NaN for now
        fr["distanceToNearest"] = np.nan

        # bedBathRatio (handle divide-by-zero)
        fr["bedBathRatio"] = fr["bedrooms"] / fr["bathrooms"] if fr["bathrooms"] > 0 else 0.0

        # ensure all features in feature_list exist in the final dict (fill sensible defaults)
        for f in feature_list:
            if f not in fr:
                # numeric defaults 0 except distances -> nan
                if "distance" in f:
                    fr[f] = np.nan
                elif f.startswith(("propertyType_", "postCode_", "suburb_")):
                    fr[f] = 0
                else:
                    fr[f] = 0

        # return a single-row DataFrame with columns in the given order
        out_df = pd.DataFrame([fr], columns=feature_list)
        return out_df

    # Feature list (same order as used during model training)
    features = [
        'landSize', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'parkingSpaces',
        'outdoorFeatures', 'indoorFeatures', 'totalFeatures',
        'propertyType_house', 'propertyType_other', 'propertyType_townhouse',
        'postCode_3103', 'postCode_3124', 'postCode_3125', 'postCode_3127', 'postCode_3128',
        'suburb_Balwyn', 'suburb_Box Hill', 'suburb_Burwood', 'suburb_Camberwell', 'suburb_Surrey Hills',
        'distanceToCBD', 'distanceToNearest', 'soldYear', 'soldMonth', 'soldWeekday', 'bedBathRatio'
    ]

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features).values
        prediction = xgb_model.predict(input_array)
        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")