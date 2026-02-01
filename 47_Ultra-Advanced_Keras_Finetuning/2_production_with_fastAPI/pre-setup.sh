# ==========================================
# STEP-BY-STEP: Setup FastAPI Environment
# ==========================================

# Create project directory
mkdir health-predictor-api
cd health-predictor-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn tensorflow joblib numpy pydantic

# Create requirements.txt for deployment
pip freeze > requirements.txt

echo "✅ Environment setup complete!"















# ==========================================
# TEST THE API LOCALLY
# ==========================================

# Start the server
uvicorn app:app --reload

# Server will start at: http://localhost:8000

# In a new terminal, test the endpoints:

# 1. Health check
curl http://localhost:8000/

# 2. Get metrics
curl http://localhost:8000/metrics

# 3. Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 0.05,
    "sex": 0.05,
    "bmi": 0.06,
    "bp": 0.02,
    "s1": -0.04,
    "s2": -0.03,
    "s3": 0.00,
    "s4": -0.03,
    "s5": 0.01,
    "s6": -0.02
  }'

# Expected response:
# {
#   "risk_probability": 0.234,
#   "risk_level": "Low",
#   "confidence": 0.532,
#   "inference_time_ms": 12.5,
#   "model_version": "1.0.0"
# }

# 4. Interactive API docs (FastAPI auto-generates!)
# Open in browser: http://localhost:8000/docs

echo "✅ API is running and ready for deployment!"