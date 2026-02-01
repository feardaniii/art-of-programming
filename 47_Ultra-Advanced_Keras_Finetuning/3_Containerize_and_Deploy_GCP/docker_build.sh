# ==========================================
# DOCKER: BUILD & TEST LOCALLY
# ==========================================

# Build the Docker image
docker build -t health-predictor-api .

# Expected output:
# [+] Building 45.2s (12/12) FINISHED
# => => naming to docker.io/library/health-predictor-api

# Run locally in container
docker run -p 8000:8000 health-predictor-api

# Test it (in new terminal)
curl http://localhost:8000/

# If it works, you're ready for cloud deployment!
echo "✅ Docker container working locally!"

# ==========================================
# GOOGLE CLOUD PLATFORM SETUP
# ==========================================

# STEP 1: Create Google Cloud Account
# Go to: https://cloud.google.com/
# - Sign up (free tier: $300 credit for 90 days)
# - Create a new project: "health-predictor"

# STEP 2: Install Google Cloud SDK
# Mac:
brew install google-cloud-sdk

# Windows: Download from https://cloud.google.com/sdk/docs/install

# Linux:
curl https://sdk.cloud.google.com | bash

# STEP 3: Initialize and Authenticate
gcloud init

# Follow prompts:
# 1. Log in with your Google account
# 2. Select your project: health-predictor
# 3. Select default region (e.g., us-central1)

# STEP 4: Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

echo "✅ Google Cloud setup complete!"

# ==========================================
# DEPLOY TO GOOGLE CLOUD RUN
# ==========================================

# Configure Docker to use Google Container Registry
gcloud auth configure-docker

# Tag your image for Google Container Registry
PROJECT_ID=$(gcloud config get-value project)
IMAGE_NAME="gcr.io/$PROJECT_ID/health-predictor-api"

docker tag health-predictor-api $IMAGE_NAME

# Push to Google Container Registry
docker push $IMAGE_NAME

# Expected output:
# The push refers to repository [gcr.io/your-project/health-predictor-api]
# latest: digest: sha256:abc123... size: 2841

echo "✅ Image pushed to Google Container Registry!"

# ==========================================
# DEPLOY TO CLOUD RUN (THE MAGIC MOMENT!)
# ==========================================

gcloud run deploy health-predictor-api \
  --image $IMAGE_NAME \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0

# You'll see:
# Deploying container to Cloud Run service [health-predictor-api]...
# ✓ Deploying... Done.
# ✓ Creating Revision...
# ✓ Routing traffic...
# Done.
# Service [health-predictor-api] revision [health-predictor-api-00001] has been deployed.
# Service URL: https://health-predictor-api-abc123-uc.a.run.app

# COPY THAT URL! This is your globally-accessible API endpoint!

# ==========================================
# TEST YOUR DEPLOYED API
# ==========================================

# Set your service URL (replace with actual URL from deployment)
SERVICE_URL="https://health-predictor-api-abc123-uc.a.run.app"

# Test health check
curl $SERVICE_URL/

# Test prediction
curl -X POST $SERVICE_URL/predict \
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
#   "inference_time_ms": 15.7,
#   "model_version": "1.0.0"
# }

echo "✅ YOUR MODEL IS NOW LIVE ON THE INTERNET!"

# ==========================================
# VIEW DEPLOYMENT DETAILS
# ==========================================

# Open Cloud Run console
echo "View your deployment:"
echo "https://console.cloud.google.com/run/detail/us-central1/health-predictor-api"

# Check logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=health-predictor-api" \
  --limit 50 \
  --format "table(timestamp,textPayload)"

# Monitor metrics
echo "View metrics at:"
echo "https://console.cloud.google.com/run/detail/us-central1/health-predictor-api/metrics"

# ==========================================
# COST ESTIMATE
# ==========================================

echo "="
echo "GOOGLE CLOUD RUN PRICING (as of 2024):"
echo "="
echo "FREE TIER (per month):"
echo "  - 2 million requests"
echo "  - 360,000 GB-seconds of memory"
echo "  - 180,000 vCPU-seconds"
echo ""
echo "PAID (after free tier):"
echo "  - $0.00002400 per request"
echo "  - $0.00000250 per GB-second"
echo "  - $0.00001000 per vCPU-second"
echo ""
echo "EXAMPLE: 10,000 predictions/month"
echo "  Cost: $0.00 (well within free tier)"
echo ""
echo "EXAMPLE: 1,000,000 predictions/month"
echo "  Cost: ~$5-15/month"
echo ""
echo "Compare to running your own server: $50-200/month"
echo "="

# ==========================================
# UPDATE DEPLOYMENT (WHEN YOU IMPROVE MODEL)
# ==========================================

# After retraining and improving your model:
# 1. Rebuild Docker image with new model
docker build -t health-predictor-api .
docker tag health-predictor-api $IMAGE_NAME
docker push $IMAGE_NAME

# 2. Deploy new version (Cloud Run handles zero-downtime rollout!)
gcloud run deploy health-predictor-api \
  --image $IMAGE_NAME \
  --platform managed \
  --region us-central1

# New revision deployed with gradual traffic shift!
echo "✅ Model updated with zero downtime!"