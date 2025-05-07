import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from Pipelines.BasePipeline import BasePipeline


class RegressionPipeline(BasePipeline):
    def __init__(self, st):
        """
        Инициализация пайплайна для задачи регрессии.

        :param st: объект Streamlit
        """
        self.title = "Regression"
        super().__init__(st, self.title)

        self.registry_models = {
            "LinearRegression": {
                "class": LinearRegression,
                "params": {},
                "name": "LinearRegression.pkl"
            },
            "GradientBoostingRegressor": {
                "class": GradientBoostingRegressor,
                "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
                "name": "GradientBoostingRegressor.pkl"
            },
            "DecisionTreeRegressor": {
                "class": DecisionTreeRegressor,
                "params": {"random_state": 42},
                "name": "DecisionTreeRegressor.pkl"
            }
        }

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("RegressionPipeline инициализирован")

    def get_data(self):
        """
        Загружает CSV-файл с данными для обучения и отображает его.

        :return: True, если файл загружен, иначе False
        """
        self.uploaded_file = self.st.file_uploader("\U0001F4C1 Загрузите CSV-файл с данными", type=["csv"])
        if self.uploaded_file:
            self.df = pd.read_csv(self.uploaded_file)
            self.all_tables = self.df.columns.tolist()
            self.st.dataframe(self.df.head())
            self.logger.info("Файл с данными успешно загружен")
            return True
        self.logger.warning("Файл с данными не был загружен")
        return False

    def train_models(self):
        """
        Обучает все выбранные регрессионные модели и сохраняет их.

        :return: None
        """
        features = [col for col in self.df.columns if col != self.target_col]
        X = self.df[features]
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        for model_name in self.selected_models:
            model_cls = self.registry_models[model_name]["class"]
            params = self.registry_models[model_name]["params"]
            model = model_cls(**params)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            joblib.dump(pipeline, f"trained_models/{self.registry_models[model_name]['name']}")
            self.logger.info("Модель %s обучена и сохранена", model_name)

            results.append({
                "Model": model_name,
                "RMSE": mean_squared_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred)
            })

        self.stat = pd.DataFrame(results)
        self.logger.info("Обучение завершено")

    def make_predict(self):
        """
        Делает предсказание на новых данных с использованием выбранной модели.

        :return: None
        """
        if "model" not in self.st.session_state:
            self.st.warning("Сначала выберите и загрузите модель.")
            self.logger.warning("Модель для предсказания не загружена")
            return

        val_file = self.st.file_uploader("\U0001F4C4 Загрузите CSV-файл для предсказания", type=["csv"], key="val")
        if val_file:
            df_val = pd.read_csv(val_file)
            if self.st.button("\U0001F52E Сделать предсказание"):
                preds = self.st.session_state.model.predict(df_val)
                df_val["Prediction"] = preds
                self.st.write("\U0001F4CA Предсказания:")
                self.st.dataframe(df_val)
                self.logger.info("Предсказания выполнены")

    def make_more_train(self):
        """
        Дообучает существующую модель на новых данных.

        :return: None
        """
        if "model" not in self.st.session_state:
            self.st.warning("Сначала выберите и загрузите модель.")
            self.logger.warning("Попытка дообучения без модели")
            return

        train_file = self.st.file_uploader("\U0001F4C4 Загрузите CSV-файл для дообучения", type=["csv"], key="retrain")
        if train_file:
            df_new = pd.read_csv(train_file)
            features = [col for col in df_new.columns if col != self.target_col]
            X_new = df_new[features]
            y_new = df_new[self.target_col]

            if self.st.button("\U0001F501 Дообучить модель"):
                model = self.st.session_state.model
                if hasattr(model, "partial_fit"):
                    model.partial_fit(X_new, y_new)
                    self.st.success("✅ Модель дообучена.")
                else:
                    model.fit(X_new, y_new)
                    self.st.warning("⚠️ Модель переобучена (старые данные не учитывались).")

                joblib.dump(model, self.st.session_state.model_title)
                self.st.success("💾 Модель сохранена.")
                self.logger.info("Модель дообучена и сохранена")
