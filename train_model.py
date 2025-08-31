import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# === Load dataset ===
# Replace with your actual dataset file
data = pd.read_csv("crop_data.csv")

# Encode categorical features
state_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

data['State'] = state_encoder.fit_transform(data['State'])
data['Crop'] = crop_encoder.fit_transform(data['Crop'])

# Features & target
X = data[['State', 'Crop', 'Year', 'Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model & encoders
pickle.dump(model, open("crop_yield_model.pkl", "wb"))
pickle.dump(state_encoder, open("state_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))

print("✅ Model and encoders saved successfully!")
