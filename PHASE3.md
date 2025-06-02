# PHASE 3: Continuous Machine Learning (CML) & Deployment

## 1. Continuous Integration & Testing
- [x] **1.1 Unit Testing with pytest**
  - [x] Test scripts for data processing, model training, and evaluation
    * Data Processing Tests (`tests/test_data.py`):
        - `test_data_preprocessing`: Ensures proper data preprocessing (normalization, missing values)
        - `test_train_test_split`: Tests data splitting functionality
    * Model Tests (`tests/test_model.py`):
        - `test_model_output_shape`: Verifies correct output shape of model predictions
        - `test_model_probability_shape`: Validates probability output shape
    * Training Tests (`tests/test_training.py`):
        - `test_train_test_split`: Ensures proper data splitting and distribution preservation
        - `test_forward_pass`: Validates model predictions on single and multiple samples
        - `test_model_training_and_evaluation`: Tests complete training pipeline and metrics
        - `test_invalid_input`: Verifies proper error handling for invalid inputs
        - `test_model_hyperparameters`: Tests model performance with different configurations:
        * Small model (50 trees, depth 3, min accuracy 0.75)
        * Medium model (100 trees, depth 6, min accuracy 0.80)
        * Large model (200 trees, depth 10, min accuracy 0.85)
        - `test_train_model_main`: Tests the main training function with configuration
  - [x] Documentation of the testing process and example test cases
    * Running Tests:
        python -m pytest
    * Example Test Cases:
        # Testing model output shapes
        def test_model_output_shape():
            model = RandomForestTrainer()
            X = sample_data.drop('Outcome', axis=1)
            predictions = model.predict(X)
            assert predictions.shape == (len(X),)
- [x] **1.2 GitHub Actions Workflows**
  - [x] CI workflows for running tests, DVC, code checks (e.g., ruff), Docker builds
  - [x] Workflow YAML files included
    * `.github/workflows/tests.yml`: main pytests file
    * `.github/workflows/codecheck.yml`: codechecking (ruff/mypy)

- [x] **1.3 Pre-commit Hooks**
  - [x] Pre-commit config and setup instructions
    * Setup Instructions:
        1. Install pre-commit: `pip install pre-commit`
        2. Install the hooks: `pre-commit install`
        3. Run manually: `pre-commit run --all-files`
        - Bypass with `git commit --no-verify -m "Bypassing pre-commit"`
        - Disable with `pre-commit uninstall`
        - Config at `.pre-commit-config.yaml`
    * Hooks included:
        - `trailing-whitespace`: Removes trailing whitespace
        - `end-of-file-fixer`: Ensures files end with a newline
        - `check-yaml`: Validates YAML syntax
        - `check-toml`: Validates TOML files
        - `check-ast`: Check valid python
        - `check-added-large-files`: Prevents large files from being committed
        - `ruff`: Python linter for code quality checks
        - `pytest`: Run pytest

## 2. Continuous Docker Building & CML
- [ ] **2.1 Docker Image Automation**
  - [ ] Automated Docker builds and pushes (GitHub Actions)
  - [ ] Dockerfile and build/push instructions for Docker Hub and GCP Artifact Registry
- [ ] **2.2 Continuous Machine Learning (CML)**
  - [ ] CML integration for automated model training on PRs
  - [ ] Example CML outputs (metrics, visualizations)
  - [ ] Setup and usage documentation

## 3. Deployment on Google Cloud Platform (GCP)
- [ ] **3.1 GCP Artifact Registry**
  - [ ] Steps for creating and pushing Docker images to GCP
- [ ] **3.2 Custom Training Job on GCP**
  - [ ] Vertex AI/Compute Engine job setup and documentation
  - [ ] Data storage in GCP bucket
- [ ] **3.3 Deploying API with FastAPI & GCP Cloud Functions**
  - [ ] FastAPI app for model predictions
  - [ ] Deployment steps and API testing instructions
- [ ] **3.4 Dockerize & Deploy Model with GCP Cloud Run**
  - [ ] Containerization and deployment steps
  - [ ] Testing and result documentation
- [ ] **3.5 Interactive UI Deployment**
  - [ ] Streamlit or Gradio app for model demonstration
  - [ ] Deployment on Hugging Face platform
  - [ ] Integration of UI deployment into GitHub Actions workflow
  - [ ] Screenshots and usage examples

## 4. Documentation & Repository Updates
- [ ] **4.1 Comprehensive README**
  - [ ] Setup, usage, and documentation for all CI/CD, CML, and deployment steps
  - [ ] Screenshots and results of deployments
- [ ] **4.2 Resource Cleanup Reminder**
  - [ ] Checklist for removing GCP resources to avoid charges

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables. Focus on automation, deployment, and clear, reproducible instructions for all steps.
