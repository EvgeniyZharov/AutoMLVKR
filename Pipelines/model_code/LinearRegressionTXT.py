def LinearRegression(model_file, **params):
    code = f'''
from sklearn.linear_model import LinearRegression
import joblib

def predict(x_data):
    model = joblib.load("{model_file}")
    y_pred = model.predict(x_data)
    print("Predicted:", y_pred)
    return y_pred
'''
    with open("LinearRegression.txt", "w", encoding="utf-8") as f:
        f.write(code)
    return "LinearRegression.txt"
