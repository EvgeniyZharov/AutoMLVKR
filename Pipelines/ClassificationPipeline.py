import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from Pipelines.BasePipeline import BasePipeline


class ClassificationPipeline(BasePipeline):
    def __init__(self, st):
        """
        Инициализация пайплайна для задачи классификации.

        :param st: объект Streamlit
        """
        self.title = "Classification"
        super().__init__(st, self.title)

        self.registry_models = {
            "LogisticRegression": {
                "class": LogisticRegression,
                "params": {"max_iter": 500},
                "name": "LogisticRegression.pkl"
            },
            "RandomForestClassifier": {
                "class": RandomForestClassifier,
                "params": {"n_estimators": 100, "random_state": 42},
                "name": "RandomForestClassifier.pkl"
            },
            "DecisionTreeClassifier": {
                "class": DecisionTreeClassifier,
                "params": {"random_state": 42},
                "name": "DecisionTreeClassifier.pkl"
            }
        }

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("ClassificationPipeline инициализирован")

    def get_data(self):
        """
        Загружает CSV-файл с данными для обучения и отображает первые строки.

        :return: True, если файл загружен, иначе False
        """
        self.uploaded_file = self.st.file_uploader("\U0001F4C1 Загрузите CSV-файл с данными", type=["csv"])
        if self.uploaded_file:
            self.df = pd.read_csv(self.uploaded_file)
            self.all_tables = self.df.columns.tolist()
            self.st.dataframe(self.df.head())
            self.logger.info("Файл данных успешно загружен")
            return True
        self.logger.warning("Файл не загружен")
        return False

    def train_models(self):
        """
        Обучает все выбранные модели классификации и сохраняет их.

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
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1-score": f1_score(y_test, y_pred, average='macro')
            })

        self.stat = pd.DataFrame(results)
        self.logger.info("Обучение моделей завершено")

    def make_predict(self):
        """
        Делает предсказание на новых данных, используя выбранную модель.

        :return: None
        """
        if "model" not in self.st.session_state:
            self.st.warning("Сначала выберите и загрузите модель.")
            self.logger.warning("Попытка предсказания без загруженной модели")
            return

        val_file = self.st.file_uploader("\U0001F4C4 Загрузите CSV-файл для предсказания", type=["csv"], key="val")
        if val_file:
            df_val = pd.read_csv(val_file)
            if self.st.button("\U0001F52E Сделать предсказание"):
                preds = self.st.session_state.model.predict(df_val)
                self.st.write("Предсказания:", preds.tolist())
                self.logger.info("Предсказания успешно выполнены")
