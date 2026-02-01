# ==========================================
# PRODUCTION MONITORING & CLIENT USAGE
# ==========================================

# Once deployed, here's how applications USE your API:

import requests
import json

# Your deployed API URL
API_URL = "https://health-predictor-api-abc123-uc.a.run.app"

# ==========================================
# EXAMPLE: HEALTHCARE APP INTEGRATION
# ==========================================

def check_patient_risk(patient_data):
    """
    Function that a healthcare app would call
    to get diabetes risk prediction.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=patient_data,
            timeout=5  # 5 second timeout
        )

        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'risk_level': result['risk_level'],
                'probability': result['risk_probability'],
                'confidence': result['confidence']
            }
        else:
            return {
                'success': False,
                'error': f"API error: {response.status_code}"
            }

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Example patient
patient = {
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
}

# Make prediction
result = check_patient_risk(patient)

if result['success']:
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Error: {result['error']}")

# ==========================================
# MONITORING WITH GOOGLE CLOUD
# ==========================================

# View real-time logs:
# gcloud logging tail "resource.type=cloud_run_revision"

# Custom metric tracking (add to app.py):
"""
from prometheus_client import Counter, Histogram
import time

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'Request latency')
PREDICTION_RISK_HIGH = Counter('predictions_high_risk', 'High risk predictions')

@app.post("/predict")
def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()  # Increment counter

    start = time.time()
    # ... prediction code ...
    REQUEST_LATENCY.observe(time.time() - start)

    if risk_level == "High":
        PREDICTION_RISK_HIGH.inc()

    return response
"""

# ==========================================
# LOAD TESTING YOUR API
# ==========================================

print("="*70)
print("LOAD TESTING")
print("="*70)

import concurrent.futures
import time

def make_prediction(i):
    """Single prediction request"""
    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=patient,
            timeout=10
        )
        latency = time.time() - start
        return {
            'success': response.status_code == 200,
            'latency': latency,
            'request_id': i
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'request_id': i}

# Simulate 100 concurrent users
num_requests = 100
print(f"Sending {num_requests} concurrent requests...")

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    results = list(executor.map(make_prediction, range(num_requests)))

total_time = time.time() - start_time

# Analyze results
successful = sum(1 for r in results if r.get('success'))
failed = num_requests - successful
latencies = [r['latency'] for r in results if r.get('success')]

print(f"\nResults:")
print(f"  Total requests: {num_requests}")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Total time: {total_time:.2f}s")
print(f"  Requests/second: {num_requests/total_time:.2f}")
print(f"  Avg latency: {sum(latencies)/len(latencies):.3f}s")
print(f"  Min latency: {min(latencies):.3f}s")
print(f"  Max latency: {max(latencies):.3f}s")
print(f"  P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")

# ==========================================
# COST MONITORING
# ==========================================

# Check current month's costs:
# gcloud billing projects describe YOUR_PROJECT_ID

# Set budget alerts:
# 1. Go to: https://console.cloud.google.com/billing/budgets
# 2. Create budget alert (e.g., $10/month threshold)
# 3. Get email when approaching limit

print("="*70)
print("MONITORING BEST PRACTICES")
print("="*70)
print("""
1. SET UP ALERTS:
   - Budget alerts ($10, $50, $100 thresholds)
   - Error rate alerts (>1% errors)
   - Latency alerts (>500ms P95)

2. DAILY CHECKS:
   - Request volume
   - Error logs
   - Latency trends
   - Cost accumulation

3. WEEKLY REVIEWS:
   - Model performance drift
   - User feedback
   - Cost efficiency
   - Optimization opportunities

4. MONTHLY AUDITS:
   - Retrain model with new data
   - Update API version
   - Review security
   - Optimize costs
""")