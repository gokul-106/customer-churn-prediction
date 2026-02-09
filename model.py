import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\task 3\customer_churn_prediction_dataset.csv")

label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])

X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
y = df['Churn']

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained - Accuracy: {accuracy:.2f}")

with open("churn_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "label_encoder": label_encoder, "features": X.columns.tolist()}, f)

app = FastAPI()

with open("churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]
    feature_names = model_data["features"]

class CustomerData(BaseModel):
    features: list[float]
    
@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        if len(data.features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, but got {len(data.features)}.")

        processed_features = np.array(data.features).reshape(1, -1)
        processed_features = scaler.transform(processed_features)
        prediction = model.predict(processed_features)

        return {
            "Churn Prediction": bool(prediction[0]),
            "Model Accuracy": round(accuracy, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
