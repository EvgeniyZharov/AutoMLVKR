def GradientBoostingRegressor(model_file, **params):
    code = f'''
from sklearn.ensemble import GradientBoostingRegressor
import joblib

def predict(x_data):
    model = joblib.load("{model_file}")
    y_pred = model.predict(x_data)
    print("Predicted:", y_pred)
    return y_pred
'''
    with open("GradientBoostingRegressor.txt", "w", encoding="utf-8") as f:
        f.write(code)
    return "GradientBoostingRegressor.txt"
