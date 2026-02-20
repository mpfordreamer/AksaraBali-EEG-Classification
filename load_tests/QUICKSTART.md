# Quick Start Guide - Load Tests

## 🚀 For Your Thesis Lecturer (Quick Demo)

### 1. Setup (One Time Only ✅)

```bash
cd load_tests
pip install -r requirements.txt 
```

### 2. Start API Server

```bash
# In a separate terminal from project root
cd api
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 3. Run Load Test with Reports

```bash
# This will generate BOTH HTML and TXT (CSV) reports as requested
cd load_tests
locust --locustfile locustfile.py --headless --users 10 --spawn-rate 2 --run-time 120s --host http://localhost:8001 --html reports/load_test_report.html --csv reports/load_test_stats
```

**Output:** Reports will be in `load_tests/reports/`:
- ✅ `load_test_report.html` - Visual HTML report
- ✅ `load_test_stats.csv` - Statistics in TXT/CSV format
- ✅ `load_test_stats_failures.csv` - Failures log
- ✅ `load_test_stats_exceptions.csv` - Exceptions log

### 4. View Results

- **HTML**: Open `reports/load_test_report.html` in browser
- **TXT/CSV**: Open `.csv` files in Excel or NotePad

---

## 📊 Understanding the Test

### What It Tests
- ✅ Health check endpoint
- ✅ EEG preprocessing workflow
- ✅ Model training
- ✅ Prediction with models
- ✅ Model management (list, download)

### Simulated Users
- 10 concurrent users
- Each user performs different operations (preprocessing, training, prediction)
- Mix of fast operations (health check) and slow operations (training)

### Test Duration
- 120 seconds (2 minutes)
- Spawns 2 new users per second until reaching 10 users

---

## 🎯 For Your Thesis

### Key Metrics to Report

From the CSV files, report:

1. **Response Times**
   - Median response time
   - 95th percentile
   - Average response time

2. **Throughput**
   - Total requests
   - Requests per second (RPS)

3. **Reliability**
   - Failure rate (%)
   - Number of errors

### Example Thesis Table

| Endpoint | Requests | Median (ms) | 95th % (ms) | Failures (%) |
|----------|----------|-------------|-------------|--------------|
| /health | 240 | 45 | 67 | 0% |
| /predict | 120 | 1250 | 2100 | 0.8% |
| /train | 10 | 85000 | 120000 | 0% |

---

## 🔄 Advanced: Stress Testing

To find the API's breaking point:

```bash
# Gradually increase to 50 users
locust --locustfile locustfile.py --headless --users 50 --spawn-rate 5 --run-time 300s --host http://localhost:8000 --html reports/stress_test.html --csv reports/stress_test
```

Watch for:
- When error rate exceeds 5%
- When response times spike
- This tells you maximum concurrent user capacity

---

## ❓ Troubleshooting

**Problem**: "Test data file not found"
```bash
# Verify files exist
ls test_data/
# Should show: baseline.mat, dataset_awal.mat, preprocessed.mat, predict.mat
```

**Problem**: "Connection refused"
```bash
# Make sure API is running
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

**Problem**: Tests timeout
- Training takes 30-120 seconds - this is normal
- If timeouts persist, increase timeout in `config.py`

---

## 📖 For More Details

See full `README.md` for:
- Detailed test scenarios
- Performance baselines
- Web UI mode (interactive)
- Customization options
- Thesis documentation guidelines
