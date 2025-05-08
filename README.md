# Diabetes Predictor

## 1. Team Information
- [x] Team Name: Triple A
- [x] Team Members (Name & Email): Annie Xu axu7@depaul.edu, Alexander Lazarov alazarov@depaul.edu, Aleksa Stefanovic astefan8@depaul.edu
- [x] Course & Section: SE489 ML ENGINEERING FOR PRODUCTION (MLOPS)

## 2. Project Overview
- [x] Brief summary of the project (2-3 sentences)
Early detection of type 2 diabetes is essential for improving patient outcomes in order to implement mitigation techniques early. We propose to build a classifier that can predict whether a patient will develop diabetes within five years using routine clinical measurements. This project will include a reproducible MLOps pipeline from data preparation, training, modeling, to classification. 
- [X] Problem statement and motivation
Type 2 diabetes affects millions of adults worldwide and can lead to many severe complications and medical problems. Current screening protocols involve blood or glucose tests and clinician judgment, which can be time consuming, costly, and is often only performed after symptoms appear. As a result, many diabetic individuals are not identified until severe damage has occurred. By applying machine learning methods to clinical measurements, we can automatically identify individuals at high risk of developing diabetes within five years, which will enable early interventions, help clinicians save time and increase accuracy, and reduce long term treatment costs.
- [x] Main objectives
- Train a ML model to predict diabetes onset using key clinical measurements
- Automate the data ingestion, preprocessing, training, evaluation steps in a reproducible 

## 3. Project Architecture Diagram
- [ ] Insert or link to your architecture diagram (e.g., draw.io, PNG, etc.)

## 4. Phase Deliverables
- [x] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions
- [x] How to set up the environment (conda/pip, requirements.txt, Docker, etc.)
    * to create environment in conda from environment.yml:
    *   conda env create -f environment.yml
    * to freeze without system prefix:
    *   conda env export --from-history | findstr -v "prefix" > environment.yml
    
    
- [x] How to run the code and reproduce results
    * train:
    *   python -m diabetes_predictor.train_model
    * predict:
    *   python -m diabetes_predictor.predict_model
- [x] data pipeline with dvc
    * install: pip install "dvc[gdrive]"
    dvc remote modify gdrive_remote gdrive_client_id <client-id> --local
    dvc remote modify gdrive_remote gdrive_client_secret <client-secret> --local
    dvc pull
## 6. Contribution Summary
- [x] Briefly describe each team member's contributions
Annie: Writing portion of README/Phase1.md
Aleksa: Researching model, writing code for implementing model
Alex: Github project setup

## 7. References
- [x] List of datasets, frameworks, and major third-party tools used
Diabetes dataset: https://www.openml.org/search?type=data&sort=runs&id=42608
Framework: scikit-learn (RandomForestClassifier, train_test_split, accuracy_score, classification_report)
Data: pandas, scipy
Development tools: Ruff (linting/formatting), MyPy (static type checking)

---

> **Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.
