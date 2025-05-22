from prometheus_client import start_http_server, Gauge
import psutil
import time
import logging

# Define Prometheus metrics
training_accuracy = Gauge('training_accuracy', 'Model Accuracy')
training_loss = Gauge('training_loss', 'Model Loss')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')

def update_metrics(acc=None, loss=None):
    if acc is not None:
        training_accuracy.set(acc)
    if loss is not None:
        training_loss.set(loss)
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    cpu_usage.set(cpu)
    memory_usage.set(mem)
    logging.info(f"CPU: {cpu}%, Memory: {mem}%")


def run_metrics_server():
    start_http_server(8000)
    while True:
        update_metrics()
        time.sleep(10)
