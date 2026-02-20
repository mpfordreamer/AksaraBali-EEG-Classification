# Script untuk menjalankan semua skenario load testing
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Load Testing - All Scenarios" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Skenario 1
Write-Host "`nRunning S1: Baseline Performance..." -ForegroundColor Yellow
locust --locustfile locustfile.py --headless --users 3 --spawn-rate 1 --run-time 60s --host http://localhost:8001 --html reports/S1_baseline_report.html --csv reports/S1_baseline_stats

Start-Sleep -Seconds 10

# Skenario 2
Write-Host "`nRunning S2: Normal Load..." -ForegroundColor Yellow
locust --locustfile locustfile.py --headless --users 10 --spawn-rate 2 --run-time 120s --host http://localhost:8001 --html reports/S2_light_load_report.html --csv reports/S2_light_load_stats

Start-Sleep -Seconds 10

# Skenario 3
Write-Host "`nRunning S3: Stress Load..." -ForegroundColor Yellow
locust --locustfile locustfile.py --headless --users 15 --spawn-rate 3 --run-time 180s --host http://localhost:8001 --html reports/S3_stress_load_report.html --csv reports/S3_stress_load_stats

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "All scenarios completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Convert semua CSV ke XLSX
Write-Host "`nConverting CSV to XLSX..." -ForegroundColor Yellow
python convert_to_xlsx.py

Write-Host "`nDone! Check reports folder for results." -ForegroundColor Green
