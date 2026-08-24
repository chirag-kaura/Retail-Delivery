# Brazil E-Commerce Logistics MLOps Pipeline 🚚📊

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-View_Here-teal?style=for-the-badge&logo=github)](https://chirag-kaura.github.io/Retail-Delivery/City_Delivery_Investment.html)

An end-to-end Machine Learning and Data Analytics project addressing delivery latency and customer churn in the Brazilian E-Commerce sector.

This repository transforms raw e-commerce data into a strategic business presentation backed by a live Machine Learning Operations (MLOps) pipeline. It identifies revenue at risk due to late deliveries and provides an interactive Random Forest model to predict latency risk in real-time.

## 🌟 Project Highlights

- **Descriptive Analytics & EDA:** Analyzed ~100k delivered orders to identify specific regional bottlenecks (e.g., Salvador, Fortaleza) where transit latency directly drives 58% of negative customer reviews.
- **Strategic Business Dashboard:** An interactive HTML dashboard presenting KPIs, drill-down charts, and an ROI simulator for optimizing fulfillment routes.
- **Machine Learning Pipeline:** A robust `scikit-learn` Pipeline predicting delivery delays (`is_late`) based on freight value, volume, weight, and geographical state mapping.
- **API Deployment:** A simulated MLOps backend using **FastAPI** to serve real-time predictions directly to the HTML frontend.
- **Data Drift Monitoring:** Background service scripts to simulate and monitor incoming distribution shifts for robust production ML.

---

## 📂 Repository Structure

```text
.
├── City_Delivery_Investment.html  # Interactive Business Presentation & UI
├── delivery_analysis_case.ipynb   # Comprehensive EDA & Statistical Testing Notebook
├── list_* (directories)           # Raw partitioned parquet dataset files
└── mlops_pipeline/                # End-to-End ML Pipeline Modular Code
    ├── models/                    # Saved .pkl model files and drift logs
    ├── train.ipynb                # Feature engineering and model training
    ├── predict.ipynb              # Jupyter-based interactive UI testing
    ├── monitor.ipynb              # Data drift monitoring simulation
    └── api_server.py              # FastAPI server serving the ML model
```

---

## 🚀 Getting Started

### 1. View the Business Dashboard
Simply click on the link below to open the dashboard in any modern web browser. No installation is required to view the charts and analysis.

👉 **[View the Live Business Dashboard](https://chirag-kaura.github.io/Retail-Delivery/City_Delivery_Investment.html)**

### 2. Run the Machine Learning API (Local)
To interact with the **Live Model Predictor** at the bottom of the HTML dashboard, start the Python backend server:

```bash
# Install required dependencies
pip install pandas scikit-learn fastapi uvicorn pydantic joblib

# Start the API Server
python mlops_pipeline/api_server.py
```
*The server will run on `http://127.0.0.1:8000` and automatically handle requests from the HTML dashboard.*

### 3. Re-Train the Model
If you wish to tweak features or hyperparameters, open and run the cells in `mlops_pipeline/train.ipynb`. This will output the new evaluation metrics and automatically overwrite the `late_delivery_model.pkl` file used by the API.

---

## 🧠 The Machine Learning Model

We predict the likelihood of an order being delayed at the time of purchase using a **Random Forest Classifier**.

- **Key Features Used:** Freight value, Product weight, Product volume, Customer state, Seller state, Purchase month, Purchase day of week.
- **Evaluation:** The model focuses on robust Recall to ensure high-risk orders are proactively identified, saving potential revenue loss and protecting seller reputation.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is open-source and available under the MIT License.
