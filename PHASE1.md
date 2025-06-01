# PHASE 1: Project Design & Model Development

## 1. Project Proposal
- [x] **1.1 Project Scope and Objectives**
  - [x] Problem statement
  Early detection of type 2 diabetes is essential for improving patient outcomes in order to implement mitigation techniques early. Manual screening by clinicians is labor intensive and can miss at-risk individuals. We propose to build a classifier that can predict whether a patient will develop diabetes within five years using routine clinical measurements.
  - [x] Project objectives and expected impact
  - Train a ML model to predict diabetes onset using key clinical measurements
  - Automate the data ingestion, preprocessing, training, evaluation steps in a reproducible MLOps framework
  - [x] Success metrics
  - Accuracy >= 75% on prediction of diabetes with the held-out test set
  - Strong F₁-score (>= 0.70): to ensure that our model has both high recall for catch those who are true diabetics cases and high precision in order to not flag healthy patients as those that are diabetic (keeping false positives low).
  - [x] 300+ word project description
  In this project, we will build a complete machine learning pipeline to predict type 2 diabetes onset using the Pima Indians Diabetes dataset. Early detection enables timely clinical interventions and can dramatically reduce long-term complications such as cardiovascular disease and neuropathy. Our model will consume eight routine clinical measurements—number of pregnancies, plasma glucose concentration, diastolic blood pressure, skinfold thickness, serum insulin, body mass index (BMI), diabetes pedigree function, and age—and output a binary risk label indicating diabetes onset within five years.
  We retrieved the dataset via the OpenML Python API. We load ARFF data into a pandas DataFrame (decoding any byte-strings) and perform a stratified 80/20 train/test split, ensuring that the proportion of diabetic vs. non‐diabetic cases remains consistent across both sets.
  For modeling, we choose a Random Forest classifier because it performs reliably on small-to-medium tabular datasets, requires only a handful of hyperparameters, and naturally guards against overfitting through bootstrap aggregation and random feature selection. For our initial model, we instantiate a RandomForestClassifier with 100 trees (n_estimators=100) and a maximum depth of 6 (max_depth=6), seeding the random generator (random_state=42) for reproducibility. Training is performed entirely on the X_train and y_train subsets.
  Once the forest is fitted, we generate class predictions for the test set and compute two core evaluation metrics. First, we print overall accuracy—the fraction of correct predictions. Then, we output a full classification report showing precision, recall, F₁-score, and support for each class. Precision tells us how often a “diabetes” prediction is correct, recall indicates how many true diabetics we identified, F₁ balances these two, and support reveals how many samples contributed to each class’s metrics.
  Finally, to preserve our trained model for later use or deployment, we serialize the fitted Random Forest. This completes a minimal but reproducible end-to-end pipeline—from raw ARFF data through model training, evaluation, and artifact persistence—providing a solid baseline for future enhancements.
  Our repository will follow MLOps best practices: scaffolded with Cookiecutter, version-controlled on GitHub, and documented in Markdown. We’ll capture dependencies in an environment.yml, enforce code style with Ruff and MyPy, and run commands with a Makefile.
- [x] **1.2 Selection of Data**
  - [x] Dataset(s) chosen and justification
  We have chosen the diabetes dataset containing eight numeric attributes: number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, age, diabetes pedigree function and a binary outcome indicating onset of diabetes within five years. This dataset includes measurements for 768 patients, a moderate size that balances training complexity and evaluation reliability, is provided in ARFF format for seamless ingestion, and is a well-established benchmark for diabetes-onset classification. We chose this because some members have experience and interest in healthcare applications of ML.
  - [x] Data source(s) and access method
  We obtained the data from OpenML which is open source. The data was downloaded as an arff file.
  - [x] Preprocessing steps
  Loaded the ARFF file, converted the resulting record array into a pandas DataFrame, and decoded any byte-strings back into Python str.
  Separated out the “class” column into y and retained the remaining eight columns as feature matrix X.
  Performed a stratified 80/20 train/test split to preserve the original positive/negative class ratio.
  - Replace placeholders or missing values with NaN
  - Drop rows with missing values
  - Convert all values to numeric
  - Normalize features (mean=0, std=1)
- [x] **1.3 Model Considerations**
  - [x] Model architecture(s) considered
  Random Forest model: This model builds many decision trees on different random subsets of the data and features, and then aggregates their predictions for the final classification.
  - [x] Rationale for model choice
  This model is the best fit since it has a smaller number of hyperparameters to tune which can deliver reliable accuracy even on a smaller dataset. It can guard against overfitting while still capturing nonlinear relationships among clinical measurements. We can also obtain clear feature-importance scores, giving clinicians actionable insights into which variables most strongly drive diabetes risk.
  - [x] Source/citation for any pre-built models
  scikit-learn was used for the random forest model
- [x] **1.4 Open-source Tools**
  - [x] Third-party package(s) selected (not PyTorch or course-used tools)
  Pandas, scipy, scikit-learn
  - [x] Brief description of how/why used
  pandas: used to load the ARFF data into a DataFrame and manipulate rows/columns for feature engineering.
  scipy: to read the .arff dataset format directly into Python.
  scikit-learn: supplies the RandomForestClassifier, train_test_split, and metric functions (accuracy_score, classification_report) for model training and evaluation.

## 2. Code Organization & Setup
- [ ] **2.1 Repository Setup**
  - [x] GitHub repo created
    https://github.com/alzlaz-learns/mlops-team-project
  - [x] Cookiecutter or similar structure used
    Cookiecutter: https://github.com/Alizadeh-DePaul/cookiecutter-mlops
- [ ] **2.2 Environment Setup**
  - [x] Python virtual environment
    conda
  - [x] requirements.txt or environment.yml
    environment.yml
  - [ ] (Optional) Google Colab setup

## 3. Version Control & Collaboration
- [x]  **3.1 Git Usage**
  - [x] Regular commits with clear messages
  - [x] Branching and pull requests
- [x] **3.2 Team Collaboration**
  - [x] Roles assigned
  Annie: Writeup
  Aleksa: Model knowledge
  Alex: Project setup/maintenance
  - [x] Code reviews and merge conflict resolution

## 4. Data Handling
- [x] **4.1 Data Preparation**
- [x] Cleaning, normalization, augmentation scripts
  Included in make_dataset.py and train_model.py
- [x] **4.2 Data Documentation**
- [x] Description of data prep process
  Loaded the ARFF file, converted the resulting record array into a pandas DataFrame, and decoded any byte-strings back into Python str.
  Separated out the “class” column into y and retained the remaining eight columns as feature matrix X.
  Performed a stratified 80/20 train/test split to preserve the original positive/negative class ratio.

## 5. Model Training
- [x] **5.1 Training Infrastructure**
  - [x] Training environment setup (e.g., Colab, GPU)
  Relied on CPU-only training using scikit-learn’s RandomForestClassifier
- [x] **5.2 Initial Training & Evaluation**
  - [x] Baseline model results
  Accuracy: 0.7467532467532467
  Classification Report:
                precision    recall  f1-score   support

            0       0.79      0.84      0.81       100
            1       0.66      0.57      0.61        54

      accuracy                           0.75       154
     macro avg       0.72      0.71      0.71       154
  weighted avg       0.74      0.75      0.74       154
  - [x] Evaluation metrics
  accuracy_score gives global correctness
  classification_report breaks performance down by class (“Diabetic” vs. “Non-Diabetic”), reporting precision, recall, and F₁-score for each label in order to see performance on each group individually (class-specific performance).

## 6. Documentation & Reporting
- [x] **6.1 Project README**
- [x] Overview, setup, replication steps, dependencies, team contributions
- [x] **6.2 Code Documentation**
- [x] Docstrings, inline comments, code style (ruff), type checking (mypy), Makefile docs
  * run mypy:
  *   mypy diabetes_predictor/
  * run ruff:
  *   ruff check .
  * fix with ruff:
  *   ruff check . --fix
  * make data       # Run the data processing script
  * make train 	    # Run the training script
  * make evaluate 	# Run the evaluation script
---

> **Checklist:** Use this as a guide. Not all items are required, but thorough documentation and reproducibility are expected.
