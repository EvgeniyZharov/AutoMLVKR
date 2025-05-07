import logging
import streamlit as st
from Pipelines.ElectronicPipeline import ElectronicPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Константы
MODES = ["train", "app"]
TASK_TYPES = {
    "Электроника (температура)": ElectronicPipeline
}

APPS = [
    {
        "title": "",
        "task": list(TASK_TYPES.keys())[0],
        "api_key": "ABC123XYZ",
        "files": [],
        "models": {"Model Pipeline": "model_rf_pipeline.joblib"}
    }
]


def check_app(api_key: str) -> dict | bool:
    """
    Проверяет, существует ли приложение по указанному API-ключу.

    :param api_key: API-ключ приложения
    :return: объект приложения или False, если не найден
    """
    logging.info("Проверка API-ключа: %s", api_key)
    matched_apps = [app for app in APPS if app["api_key"] == api_key]
    if matched_apps:
        logging.info("Найдено приложение: %s", matched_apps[0]["title"])
        return matched_apps[0]
    else:
        logging.warning("Приложение с указанным ключом не найдено.")
        st.warning("Такого приложения нет.")
        return False


def run_system() -> None:
    """
    Главная функция запуска системы. Определяет режим работы (обучение или приложение),
    и запускает соответствующий pipeline.

    :return: None
    """
    logging.info("Запуск системы AutoML")
    st.set_page_config(page_title="AutoML System")

    query_params = st.query_params
    status = query_params.get("status", "user")
    mode = query_params.get("mode", "train")
    api_key = query_params.get("api_key", "None")

    logging.info("Режим: %s | Статус: %s", mode, status)

    if mode == "train":
        task_type = st.selectbox("Выберите тип задачи", list(TASK_TYPES.keys()))
        if "pipeline" not in st.session_state:
            logging.info("Создание pipeline для задачи: %s", task_type)
            st.session_state.pipeline = TASK_TYPES[task_type](st)
        st.session_state.pipeline.run()

    elif mode == "app":
        result = check_app(api_key=api_key)
        if result:
            if "pipeline" not in st.session_state:
                st.session_state.app = result
                logging.info("Создание pipeline для приложения: %s", result["title"])
                st.session_state.pipeline = TASK_TYPES[result["task"]](st)
            st.session_state.pipeline.run_app()


# Запуск при инициализации
run_system()
