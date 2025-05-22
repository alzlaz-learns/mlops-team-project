# 📊 Monitoring Setup (Prometheus + Python)

This folder provides a real-time monitoring system for our ML training pipeline using **Prometheus** and Python’s `prometheus_client`.

---

## 🎯 What It Does

- Tracks **model performance**: `training_accuracy`, `training_loss`
- Monitors **system resources**: `cpu_usage_percent`, `memory_usage_percent`
- Exposes metrics via HTTP for Prometheus to scrape
- Compatible with Prometheus/Grafana dashboards

---

## Metrics Exposed:

| Metric Name            | Description                               |
| ---------------------- | ----------------------------------------- |
| `training_accuracy`    | Final model accuracy after training       |
| `training_loss`        | Training loss (placeholder value for now) |
| `cpu_usage_percent`    | System CPU usage in real time             |
| `memory_usage_percent` | System memory usage in real time          |

---

## 📁 Folder Contents

| File                | Description                                    |
|---------------------|------------------------------------------------|
| `metrics_logger.py` | Python script that exposes metrics             |
| `prometheus.yml`    | Prometheus scrape configuration file           |
| `README.md`         | You are here – full setup and usage guide      |

---

## 📦 Installation Requirements
Step 1: Download Prometheus

Install required Python packages:
`bash`
pip install prometheus_client psutil

Step 2: Copy this file: `monitoring/prometheus.yml` into your Prometheus folder, or point to it with:

./prometheus --config.file=monitoring/prometheus.yml

Step 3: Integrate with `train_model.py`
