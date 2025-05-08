# Diabetes Predictor

## 1. Team Information
- [x] Team Name: Triple A
- [ ] Team Members (Name & Email): Annie Xu, axu7@depaul.edu, Alexander Lazarov alazarov@depaul.edu
- [x] Course & Section: SE489 ML ENGINEERING FOR PRODUCTION (MLOPS)

## 2. Project Overview
- [ ] Brief summary of the project (2-3 sentences)
- [ ] Problem statement and motivation
- [ ] Main objectives

## 3. Project Architecture Diagram
- [ ] Insert or link to your architecture diagram (e.g., draw.io, PNG, etc.)

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
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

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions

## 7. References
- [ ] List of datasets, frameworks, and major third-party tools used

---

> **Tip:** Keep this README updated as your project evolves. Link to each phase deliverable and update the architecture diagram as your pipeline matures.
