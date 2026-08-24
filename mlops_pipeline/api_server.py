import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = r"E:\Praxis\TERM2\DAS\retail_parquet"
MODEL_PATH = os.path.join(BASE_DIR, "mlops_pipeline", "models", "late_delivery_model.pkl")

# Load model globally when server starts
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class OrderData(BaseModel):
    revenue: float
    freight: float
    items_count: int
    product_weight_g: float
    product_volume: float
    customer_state: str
    seller_state: str
    purchase_month: int
    purchase_dayofweek: int

@app.get("/")
def read_root():
    return {"message": "The MLOps API Server is running perfectly! Please open 'City_Delivery_Investment.html' in your browser to use the interactive dashboard."}

@app.post("/predict")
def predict(order: OrderData):
    if model is None:
        return {"error": "Model not loaded properly."}
        
    df = pd.DataFrame([order.dict()])
    prob = model.predict_proba(df)[0][1]
    is_late = int(model.predict(df)[0])
    
    return {
        "is_late": is_late,
        "probability": float(prob),
        "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low",
        "action": "Subsidize Expedited Shipping" if is_late == 1 else "Standard Routing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
