def DecisionTreeRegressor(model_file, **params):
    code = f'''
from sklearn.tree import DecisionTreeRegressor
import joblib

def predict(x_data):
    model = joblib.load("{model_file}")
    y_pred = model.predict(x_data)
    print("Predicted:", y_pred)
    return y_pred
'''
    with open("DecisionTreeRegressor.txt", "w", encoding="utf-8") as f:
        f.write(code)
    return "DecisionTreeRegressor.txt"
