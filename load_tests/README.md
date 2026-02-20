# Load Tests for Aksara Bali EEG Classification API

Professional-grade load testing suite using [Locust](https://locust.io/) for the Aksara Bali EEG Classification API. This test suite is designed for thesis documentation and production readiness evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Test Data](#test-data)
- [Test Scenarios](#test-scenarios)
- [Running Tests](#running-tests)
- [Understanding Results](#understanding-results)
- [Performance Baselines](#performance-baselines)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Thesis Documentation](#thesis-documentation)

---

## 🎯 Overview

This load testing suite simulates realistic user behaviors and workloads for your EEG classification API. It measures:

- **Response times** for each endpoint
- **Throughput** (requests per second)
- **Error rates** under load
- **System behavior** with concurrent users
- **Performance degradation** under stress

### Tested Endpoints

- `GET /health` - Health check
- `POST /preprocess` - EEG data preprocessing
- `POST /preprocess/download` - Download preprocessed data
- `POST /train` - Model training
- `POST /train/save` - Save trained model
- `GET /train/download` - Download model file
- `GET /train/plot/confusion-matrix` - Get confusion matrix visualization
- `POST /predict` - Make predictions
- `GET /models/list` - List available models
- `DELETE /models/delete/{name}` - Delete models

---

## 🚀 Setup

### Prerequisites

- **Python 3.10+** (same as your API)
- **Running API server** on `http://localhost:8000` (or configure `API_BASE_URL` in `config.py`)
- **Test data files** (automatically copied from `tests/test_data/`)

### Installation

1. Navigate to the load tests directory:
   ```bash
   cd load_tests
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   locust --version
   ```

### Start Your API Server

**Before running load tests**, make sure your API is running:

```bash
# In a separate terminal, from project root
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify API is running:
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

---

## 📊 Test Data

The load tests use the following data files from `test_data/`:

| File | Size | Purpose |
|------|------|---------|
| `baseline.mat` | 344 KB | Baseline EEG signal for preprocessing |
| `dataset_awal.mat` | 1.8 MB | Training EEG data for preprocessing |
| `preprocessed.mat` | 32 KB | Preprocessed features for training |
| `predict.mat` | 32 KB | Test data for predictions |

### Using Your Own Data

To test with your thesis dataset:

1. Copy your `.mat` files to `load_tests/test_data/`
2. Update `config.py` to reference your files:
   ```python
   TEST_FILES = {
       "baseline": os.path.join(TEST_DATA_DIR, "your_baseline.mat"),
       "dataset_awal": os.path.join(TEST_DATA_DIR, "your_training.mat"),
       # ... etc
   }
   ```

---

## 👥 Test Scenarios

### 1. HealthCheckUser (Weight: 2)
- **Behavior**: Only checks `/health` endpoint
- **Use case**: Simulates monitoring systems, load balancers
- **Frequency**: High (2x weight)

### 2. PreprocessingUser (Weight: 1)
- **Behavior**: Uploads and preprocesses EEG data
- **Endpoints**: `/preprocess`, `/preprocess/download`
- **Use case**: Researchers preprocessing new datasets
- **Frequency**: Low (occasional operation)

### 3. TrainingUser (Weight: 1)
- **Behavior**: Trains models, saves, downloads
- **Endpoints**: `/train`, `/train/save`, `/train/download`, `/train/plot/confusion-matrix`
- **Use case**: Model experimentation
- **Frequency**: Low (computationally expensive)
- **Wait time**: 5-15 seconds between actions

### 4. PredictionUser (Weight: 3)
- **Behavior**: Makes predictions with models
- **Endpoints**: `/predict` (default and named models)
- **Use case**: Main production use case
- **Frequency**: High (3x weight - most common operation)

### 5. ModelManagementUser (Weight: 1)
- **Behavior**: Lists and manages models
- **Endpoints**: `/models/list`, `/models/delete/{name}`
- **Use case**: Browsing and managing models
- **Frequency**: Medium

### 6. MixedWorkflowUser (Weight: 2)
- **Behavior**: Complete end-to-end workflows
- **Endpoints**: Mix of all above
- **Use case**: Realistic full pipeline usage
- **Frequency**: Medium

---

## 🏃 Running Tests

### 1. Web UI Mode (Recommended)

Best for interactive testing and visualization:

```bash
locust --locustfile locustfile.py --host http://localhost:8000
```

Then open **http://localhost:8089** in your browser.

**Steps:**
1. Enter number of users: `10`
2. Enter spawn rate: `2` (users/second)
3. Click "Start swarming"
4. Monitor real-time charts
5. Download reports when done

### 2. Headless Mode with Reports

For automated testing and thesis documentation:

```bash
# Generate both HTML and CSV reports
locust --locustfile locustfile.py --headless \
    --users 10 --spawn-rate 2 --run-time 120s \
    --host http://localhost:8000 \
    --html reports/load_test_report.html \
    --csv reports/load_test_stats
```

**Output files:**
- `reports/load_test_report.html` - Visual report with charts
- `reports/load_test_stats.csv` - Request statistics (for Excel/analysis)
- `reports/load_test_stats_failures.csv` - Failure log
- `reports/load_test_stats_exceptions.csv` - Exception details

### 3. Test Specific User Type

Test only one user behavior:

```bash
locust --locustfile locustfile.py --headless \
    --users 5 --spawn-rate 1 --run-time 60s \
    --host http://localhost:8000 --user HealthCheckUser \
    --html reports/health_check_test.html
```

Available user types:
- `HealthCheckUser`
- `PreprocessingUser`
- `TrainingUser`
- `PredictionUser`
- `ModelManagementUser`
- `MixedWorkflowUser`

### 4. Stress Testing

Find the breaking point of your API:

```bash
# Gradually increase load
locust --locustfile locustfile.py --headless \
    --users 50 --spawn-rate 5 --run-time 300s \
    --host http://localhost:8000 \
    --html reports/stress_test_report.html \
    --csv reports/stress_test_stats
```

Monitor for:
- Response time degradation
- Error rate increases
- When system becomes unstable

---

## 📈 Understanding Results

### Key Metrics

#### 1. Response Time
- **Median**: 50th percentile (typical response time)
- **95th percentile**: 95% of requests faster than this
- **Average**: Mean response time
- **Min/Max**: Fastest and slowest requests

#### 2. Requests Per Second (RPS)
- Total throughput of the API
- Higher is better (but watch error rates)

#### 3. Failure Rate
- Percentage of failed requests
- **Good**: < 1%
- **Acceptable**: 1-5%
- **Poor**: > 5%

### Reading HTML Reports

The HTML report includes:

1. **Statistics Table**
   - All endpoints listed
   - Response times and failure rates
   - RPS and request counts

2. **Charts Tab**
   - Response time over time
   - RPS over time
   - Number of users over time

3. **Failures Tab**
   - Detailed error messages
   - Frequency of each error type

4. **Exceptions Tab**
   - Python exceptions (if any)

### Reading CSV Reports

Three CSV files are generated:

1. **`*_stats.csv`**: Main statistics
   ```
   Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,...
   GET,/health,1000,0,45,48.5,...
   POST,/predict,500,2,1250,1340.2,...
   ```

2. **`*_stats_failures.csv`**: Errors
   ```
   Method,Name,Error,Occurrences
   POST,/predict,"HTTP 400: Invalid data",2
   ```

3. **`*_stats_exceptions.csv`**: Exceptions
   ```
   Count,Message,Traceback,Nodes
   1,"Connection timeout","...",worker_1
   ```

---

## ⚡ Performance Baselines

Expected response times for your API:

| Endpoint | Expected Time | Acceptable Max |
|----------|---------------|----------------|
| `/health` | < 100 ms | 200 ms |
| `/preprocess` | 2-5 seconds | 10 seconds |
| `/train` | 30-120 seconds | 180 seconds |
| `/predict` | 1-3 seconds | 5 seconds |
| `/models/list` | < 500 ms | 1 second |

**Note**: These are baselines for the test dataset. Your thesis data may vary.

### Recommended Concurrent Users

Based on typical hardware:

- **Light load**: 5-10 concurrent users
- **Medium load**: 10-25 concurrent users
- **Heavy load**: 25-50 concurrent users
- **Stress test**: 50-100+ concurrent users

---

## ⚙️ Customization

### Adjust Load Parameters

Edit `config.py`:

```python
# Wait times between user actions (milliseconds)
MIN_WAIT = 1000  # 1 second
MAX_WAIT = 5000  # 5 seconds

# API timeout
API_TIMEOUT = 180  # 3 minutes

# Performance thresholds
THRESHOLDS = {
    "health_max_response": 100,
    "predict_max_response": 5000,
    # ... customize as needed
}
```

### Change User Distribution

Edit `locustfile.py`:

```python
class PredictionUser(HttpUser):
    weight = 5  # Increase from 3 to make predictions more frequent
    # ...
```

### Add New Test Scenarios

Create new task file in `tasks/` directory following existing patterns.

---

## 🔧 Troubleshooting

### Problem: "Test data file not found"

**Solution**: Ensure test data files are in `load_tests/test_data/`:
```bash
ls test_data/
# Should show: baseline.mat, dataset_awal.mat, preprocessed.mat, predict.mat
```

### Problem: "Connection refused"

**Solution**: Make sure API server is running:
```bash
# Check if API is accessible
curl http://localhost:8000/health
```

### Problem: High error rates

**Possible causes:**
1. API server overloaded (reduce concurrent users)
2. Timeout too short (increase `API_TIMEOUT` in config.py)
3. Test data incompatible with API version

**Debug steps:**
```bash
# Run with verbose logging
locust -f locustfile.py --host http://localhost:8000 --loglevel DEBUG
```

### Problem: Slow training requests timing out

**Solution**: Increase timeout in `config.py`:
```python
API_TIMEOUT = 300  # 5 minutes for very large datasets
```

---

## 📚 Thesis Documentation

### Including Results in Your Thesis

#### 1. Performance Evaluation Chapter

**Screenshots to include:**
- Locust web UI dashboard showing real-time metrics
- HTML report statistics table
- Response time charts over test duration
- RPS (throughput) charts

**Metrics to report:**
```
Under light load (10 concurrent users):
- Average response time: XX ms
- 95th percentile: XX ms
- Throughput: XX requests/second
- Error rate: X.X%

Under heavy load (50 concurrent users):
- Average response time: XX ms
- 95th percentile: XX ms
- Throughput: XX requests/second
- Error rate: X.X%
```

#### 2. Scalability Analysis

Run tests with increasing users:

```bash
# Test 1: 10 users
locust -f locustfile.py --headless --users 10 --spawn-rate 2 --run-time 120s \
    --host http://localhost:8000 --csv results/test_10users

# Test 2: 25 users
locust -f locustfile.py --headless --users 25 --spawn-rate 5 --run-time 120s \
    --host http://localhost:8000 --csv results/test_25users

# Test 3: 50 users
locust -f locustfile.py --headless --users 50 --spawn-rate 5 --run-time 120s \
    --host http://localhost:8000 --csv results/test_50users
```

Create a table:

| Concurrent Users | Avg Response Time | RPS | Error Rate |
|------------------|-------------------|-----|------------|
| 10 | XX ms | XX | X% |
| 25 | XX ms | XX | X% |
| 50 | XX ms | XX | X% |

#### 3. System Requirements Recommendations

Based on stress test results:

```
Recommended Production Configuration:
- CPU: X cores minimum
- RAM: X GB minimum
- Concurrent user capacity: X users
- Expected throughput: X requests/second
```

#### 4. Test Methodology Section

Include:
- Load testing tool: Locust v2.20+
- Test scenarios: List the 6 user types
- Test duration: XX minutes per scenario
- Test data: Description and sizes
- Performance baselines: Table of expected times

### Example Results Section

```markdown
## 5.3 Performance Testing Results

Load tests were conducted using Locust, an open-source load testing tool. 
Six user behavior types were defined to simulate realistic API usage patterns:

1. Light health checks (20% of traffic)
2. Data preprocessing workflows (10%)
3. Model training operations (10%)
4. Prediction requests (30%)
5. Model management (10%)
6. Mixed end-to-end workflows (20%)

### 5.3.1 Light Load Performance

Under light load conditions (10 concurrent users), the system demonstrated 
excellent performance:

- Health endpoint: median 45ms, 95th percentile 62ms
- Prediction endpoint: median 1.2s, 95th percentile 2.1s
- Training endpoint: median 85s, 95th percentile 120s
- Overall throughput: 8.5 requests/second
- Error rate: 0.1%

[Insert screenshot of Locust dashboard here]

### 5.3.2 Stress Test Results

The system remained stable up to 35 concurrent users, after which:
- Response times increased by 150%
- Error rate rose to 8%
- Throughput plateaued at 12 requests/second

[Insert response time degradation chart here]

These results indicate the current deployment can handle approximately 
30-35 concurrent users before requiring horizontal scaling.
```

---

## 📝 Quick Reference

### Most Common Commands

```bash
# Interactive testing (Web UI)
locust -f locustfile.py --host http://localhost:8000

# Automated test with reports
locust -f locustfile.py --headless --users 10 --spawn-rate 2 --run-time 60s \
    --host http://localhost:8000 \
    --html reports/report.html --csv reports/stats

# Stress test
locust -f locustfile.py --headless --users 50 --spawn-rate 5 --run-time 300s \
    --host http://localhost:8000 \
    --html reports/stress_report.html

# Test single user type
locust -f locustfile.py --headless --users 5 --spawn-rate 1 --run-time 30s \
    --host http://localhost:8000 --user PredictionUser
```

### Environment Variables

Override configuration via environment variables:

```bash
# Windows
set API_BASE_URL=http://192.168.1.100:8000
set API_TIMEOUT=300
locust -f locustfile.py

# Linux/Mac
API_BASE_URL=http://192.168.1.100:8000 API_TIMEOUT=300 locust -f locustfile.py
```

---

## 📞 Support

For thesis-related questions or load testing customization:
- Review Locust documentation: https://docs.locust.io/
- Check API logs for detailed error messages
- Adjust timeouts and user counts based on your hardware

---

## 📄 License

Part of the Aksara Bali EEG Classification project.

© I Dewa Gede Mahesta Parawangsa  
[LinkedIn](https://www.linkedin.com/in/demahesta)

---

**Happy Load Testing! 🚀**
