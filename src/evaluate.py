# src/evaluate.py
import joblib
import yaml
from DeepEval import evaluate_model

def load_config():
    with open("src/config.yaml", "r") as file:
        return yaml.safe_load(file)

def evaluate():
    # Load configuration
    config = load_config()
    
    # Load model and test data
    model = joblib.load('model.joblib')
    
    # Example prediction (replace with actual test set evaluation)
    # Assuming a method to get X_test, y_test
    predictions = model.predict(X_test)
    results = evaluate_model(predictions, y_test, metrics=config["metrics"])

    print("Evaluation Results:", results)

if __name__ == "__main__":
    evaluate()
