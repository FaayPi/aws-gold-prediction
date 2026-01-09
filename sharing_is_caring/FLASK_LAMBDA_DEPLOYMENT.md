# Deploying Flask Apps with AWS Lambda & API Gateway
## A Serverless Approach for ML Model Serving

---

## Overview

Instead of managing servers, we deploy our Flask application as a **serverless function** using AWS Lambda and expose it via API Gateway. This approach is cost-effective, scalable, and requires zero server management.

---

## Architecture: The Big Picture

```
┌─────────────┐
│     User    │
│  (Browser)  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐         ┌──────────────┐         ┌─────────────┐
│   Lightsail     │────────▶│ API Gateway  │────────▶│   Lambda    │
│  (Flask Web App)│         │   (HTTP API) │         │ (Flask API) │
│  Port 8080/5000 │         └──────────────┘         └──────┬──────┘
└─────────────────┘                                         │
                                                            ▼
                                                     ┌─────────────┐
                                                     │     S3      │
                                                     │(Predictions)│
                                                     └─────────────┘
```

**Flow:**
1. User opens web app hosted on Lightsail
2. Web app makes HTTP request to API Gateway
3. API Gateway triggers Lambda function
4. Lambda function reads predictions from S3
5. Response flows back: Lambda → API Gateway → Web App → Client

---

## Why This Approach?

### ✅ **Advantages:**
- **No Server Management:** AWS handles infrastructure
- **Pay-per-Use:** Only pay for actual invocations
- **Simple Deployment:** Just zip your code and upload
- **Built-in Security:** API Gateway handles authentication/authorization

---

## Key Components

### 1. **AWS Lambda**
- Runs your Flask application as a function
- Charges only for compute time used

### 2. **API Gateway**
- Creates a RESTful HTTP API endpoint
- Handles routing, authentication, rate limiting
- Provides a public URL for your Lambda function (lambda itself does not provide public URL)

### 3. **S3 (Simple Storage Service)**
- Stores pre-computed predictions as JSON files

### 4. **AWS Lightsail** (Web App Hosting)
- Hosts the Flask web application (frontend/UI)
- Simplified VPS alternative to EC2
- Fixed, predictable pricing
- Pre-configured instances for easy deployment

---

## How It Works: Step by Step

### Step 1: Prepare Your Flask Apps
- **Flask API for Lambda**: Create a Flask app that reads predictions from S3 (backend)
- **Flask Web App for Lightsail**: Create a Flask web app (frontend)
- API Gateway makes lambda available: Web app (Lightsail) → API Gateway → Lambda (API)

### Step 2: Package Your Code
- Create a deployment package (ZIP file)
- Include: Flask app, dependencies, `lambda_handler.py`
- Keep it under 50MB (unzipped) for direct upload

### Step 3: Deploy to Lambda
- Upload ZIP file to Lambda function and set configurations 

### Step 4: Create API Gateway
- Create REST API or HTTP API
- Integrate with Lambda function
- Deploy to stage (e.g., `prod`)

### Step 5: Deploy Web App to Lightsail
- Create a Lightsail instance (choose Linux/Unix platform)
- Connect via SSH and install Python, Flask, and dependencies
- Upload your Flask web app code (the one that calls API Gateway)
- Configure the web app to use the API Gateway URL
- Set up a static IP and configure the instance to run on port 8080 or 5000
- Your web app is now accessible via the Lightsail public IP!

### Step 6: Test & Use
- **API Gateway + Lambda** (Backend - Serverless):
  - API Gateway provides a public URL for the API:
    ```
    https://abc123.execute-api.region.amazonaws.com/prod/predict
    ```
  - Lambda runs **on-demand** (no server running 24/7)

- **Lightsail** (Frontend - Server-based):
  - Lightsail provides a public IP for your web app:
    ```
    http://your-lightsail-ip:8080
    ```
  - Web app runs on a **virtual server** (runs 24/7)

---

## My Implementation Details

### **Complete Architecture:**
- **Lightsail**: Hosts Flask web app (frontend) - displays predictions to users
- **Lambda + API Gateway**: Serves predictions API (backend)
- **S3**: Stores pre-computed predictions (generated daily)

### **Benefits of This Pattern:**
- ✅ Cost-effective (Lambda only runs when called)
- ✅ Separation of concerns: Frontend (Lightsail) + Backend (Lambda)
- ✅ Easier Maintenance: Updates to backend don't require frontend changes and vice versa

---

## Lightsail vs EC2: Choosing the Right Hosting

### **Why Lightsail for the Web App?**

**Lightsail** is AWS's simplified VPS (Virtual Private Server) solution, perfect for hosting simple web applications like our Flask frontend.

#### **Lightsail Advantages:**
- ✅ **Simplified Setup**: Pre-configured instances, ready to use
- ✅ **Fixed Pricing**: Predictable monthly costs ($3.50-$40/month)
- ✅ **Easy Management**: Simple console, less configuration needed
- ✅ **Perfect for Simple Apps**: Ideal for web apps, blogs, small APIs
- ✅ **Includes**: Static IP, DNS management, load balancer (optional)


### **Comparison: Lightsail vs EC2**

| Feature | Lightsail | EC2 |
|---------|-----------|-----|
| **Setup Complexity** | Simple, pre-configured | More configuration needed |
| **Pricing Model** | Fixed monthly price | Pay-per-use (can be cheaper or more expensive) |
| **Cost (1GB RAM, 1 vCPU)** | ~$5/month (fixed) | ~$7-10/month (on-demand, 24/7) |
| **Storage** | Included in price | Separate EBS volumes |
| **Best For** | Simple web apps, blogs | Complex applications, enterprise |


---

**Questions?** 🚀

