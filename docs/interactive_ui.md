# Interactive UI Deployment - Diabetes Predictor

**Deployment Date:** June 05, 2025  
**Platform:** Hugging Face Spaces  
**Interface Type:** Gradio  
**Directory:** `diabetes-predictor-ui/`

---

## Project Summary

This interactive UI showcases our trained diabetes prediction model using a user-friendly Gradio web interface. The app accepts medical input features and returns a prediction indicating the likelihood of a diabetes diagnosis.

---

## How It Works

- The model was trained using a Random Forest classifier on a cleaned version of the diabetes dataset.
- Input features include glucose level, BMI, age, blood pressure, and other medical indicators.
- The app loads the trained model (`model.joblib`) and uses it to generate real-time predictions.

---

## File Structure

```bash
diabetes-predictor-ui/
├── app.py                  # Gradio app script
├── requirements.txt        # Hugging Face-specific dependencies
├── model.joblib            # Trained model file
```

---

## Deployment on Hugging Face

### Steps

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space → choose `Gradio` as the SDK
3. Upload the contents of `diabetes-predictor-ui/`
4. Set the Python version and runtime if needed
5. Click “Deploy”

Hugging Face will install the dependencies from `requirements.txt` and run `app.py`.

---

## Example Interface

> Add a screenshot in the repo at `docs/screenshots/interactive-ui.png` and it will appear below.

```
![Gradio UI Screenshot](../docs/screenshots/interactive-ui.png)
```

---

## Notes

- The model file must match the filename used in `app.py`
- Ensure `requirements.txt` includes `gradio`, `scikit-learn`, and `joblib`
- If the app doesn't start, check the Space logs for errors and confirm model path

---

