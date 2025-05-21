# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## 1. Containerization
- [x] **1.1 Dockerfile**
  - [x] Dockerfile created and tested
  - [x] Instructions for building and running the container
- [x] **1.2 Environment Consistency**
  - [x] All dependencies included in the container

## 2. Monitoring & Debugging

- [ ] **2.1 Debugging Practices**
  - [x] Debugging tools used (e.g., pdb)
    * pdb
  - [ ] Example debugging scenarios and solutions

## 3. Profiling & Optimization
- [ ] **3.1 Profiling Scripts**
  - [ ] cProfile, PyTorch Profiler, or similar used
  - [ ] Profiling results and optimizations documented

## 4. Experiment Management & Tracking
- [x] **4.1 Experiment Tracking Tools**
  - [x] MLflow, Weights & Biases, or similar integrated
      * Experiment Tracking with MLflow
      * We use MLflow to track experiments and manage model data
      * install mlflow
      *   pip install mlflow
  - [x] Logging of metrics, parameters, and models
      * ![alt text](image-1.png)
      * ![alt text](image.png)
  - [x] Instructions for visualizing and comparing runs
      * visualize outputs on local machine:
          *   mlflow ui
          *   open browser go to:
          *     http://localhost:5000/
          *     diabetes-prediction -> select run -> Model metrics for visualizations

## 5. Application & Experiment Logging
- [ ] **5.1 Logging Setup**
  - [ ] logger and/or rich integrated
  - [ ] Example log entries and their meaning

## 6. Configuration Management
- [ ] **6.1 Hydra or Similar**
  - [ ] Configuration files created
  - [ ] Example of running experiments with different configs

## 7. Documentation & Repository Updates
- [ ] **7.1 Updated README**
  - [ ] Instructions for all new tools and processes
  - [ ] All scripts and configs included in repo

---

> **Checklist:** Use this as a guide for documenting your Phase 2 deliverables. Focus on operational robustness, reproducibility, and clear instructions for all tools and processes.

