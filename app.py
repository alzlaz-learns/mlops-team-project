import gradio as gr
import joblib

# Load model
model = joblib.load("model.joblib")  # we'll copy this into the same folder

# Predict function
def predict_diabetes(
    pregnancies: float,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: float
) -> str:
    features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(features)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"

# Gradio UI
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Risk Predictor",
    description="Input patient data to predict diabetes risk using a trained model."
)

if __name__ == "__main__":
    iface.launch()
