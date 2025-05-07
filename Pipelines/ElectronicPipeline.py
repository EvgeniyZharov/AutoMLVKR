import logging
from io import StringIO
from Pipelines.BasePipeline import BasePipeline


class ElectronicPipeline(BasePipeline):
    def __init__(self, st):
        """
        Инициализация пайплайна для задачи временного ряда (электроника).

        :param st: объект Streamlit
        """
        import pandas as pd
        from Pipelines.registry_models import registry_models

        self.title = "Обучение модели для задачи временного ряда температуры (электроника)"
        super().__init__(st, self.title)

        self.registry_models = registry_models
        self.n_lags = 5

        # Пример данных компонентов
        components_data = """
        Материал полупроводника,Класс прибора,Тип проводимости,"Концентрация примесей (допинг), см⁻³","Длина канала, мкм","Ширина канала, мкм","Напряжение питания, В","Напряжение на затворе, В","Нормальное состояние, К",ID компонента
        Si,гражд. н.,n-типа,2.50E+17,1,5,0,0,398,1
        Si,гражд. н.,n-типа,2.50E+17,1,10,0,0,398,2
        """
        self.example_components = pd.read_csv(StringIO(components_data))

        # Пример температурных данных
        temperature_data = """
        ID компонента,Время,Температура
        1,0.0,98.81
        1,0.10,99.86
        1,0.21,103.03
        1,0.31,103.78
        """
        self.example_temperature = pd.read_csv(StringIO(temperature_data))

        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализирован ElectronicPipeline")

    def build_dl_model(self, model_type="LSTM", input_shape=(5, 1), units=64):
        """
        Строит и возвращает нейронную модель на основе заданного типа.

        :param model_type: тип модели ('LSTM', 'GRU', 'CNN_LSTM' и т.д.)
        :param input_shape: форма входных данных
        :param units: количество нейронов в скрытом слое
        :return: скомпилированная модель Keras
        """
        from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Bidirectional, Conv1D, MaxPooling1D, ConvLSTM2D
        from tensorflow.keras.layers import BatchNormalization, Flatten
        from tensorflow.keras.models import Sequential

        model = Sequential()

        if model_type == "LSTM":
            model.add(LSTM(units, input_shape=input_shape))
        elif model_type == "LSTM_GRU":
            model.add(LSTM(units, input_shape=input_shape))
        elif model_type == "CNN_LSTM":
            model.add(Conv1D(filters=units, kernel_size=3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(50))
        elif model_type == "ConvLSTM":
            model.add(ConvLSTM2D(filters=units, kernel_size=(2, 2), activation='relu',
                                 input_shape=(5, 1), return_sequences=False, padding='same'))
            model.add(BatchNormalization())
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
        elif model_type == "GRU":
            model.add(GRU(units, input_shape=input_shape))
        elif model_type == "SimpleRNN":
            model.add(SimpleRNN(units, input_shape=input_shape))
        elif model_type == "BidirectionalLSTM":
            model.add(Bidirectional(LSTM(units), input_shape=input_shape))
        elif model_type == "BidirectionalGRU":
            model.add(Bidirectional(GRU(units), input_shape=input_shape))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")

        self.logger.info("Собрана модель типа %s", model_type)
        return model

    def get_data(self):
        """
        Загружает данные компонентов и температур, объединяет и сохраняет таблицу в состоянии.

        :return: True если оба файла загружены, иначе False
        """
        self.logger.info("Ожидание загрузки CSV-файлов от пользователя")
        text_1 = """Загрузите CSV-файл с компонентами..."""
        text_2 = """Загрузите CSV-файл с температурой..."""

        self.components_file = self.st.file_uploader(text_1, type=["csv"])
        self.st.dataframe(self.example_components)
        self.temperature_file = self.st.file_uploader(text_2, type=["csv"])
        self.st.dataframe(self.example_temperature)

        if self.components_file and self.temperature_file:
            self.df = self.merge(self.components_file, self.temperature_file)
            self.all_tables = self.df.columns.tolist()
            self.logger.info("Файлы успешно загружены и объединены")
            return True
        else:
            self.logger.warning("Файлы не загружены")
            return False

    def merge(self, file1, file2):
        """
        Объединяет два CSV-файла по ID компонента.

        :param file1: файл с компонентами
        :param file2: файл с температурой
        :return: объединённый DataFrame
        """
        import pandas as pd

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        merged_df = df2.merge(df1, on="ID компонента", how="left")

        self.st.write("📄 Предпросмотр данных:")
        self.st.dataframe(merged_df.head())
        merged_df.to_csv("test_temp.csv", index=False)

        self.logger.info("Данные успешно объединены и сохранены в test_temp.csv")
        return merged_df
