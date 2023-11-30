import dill
import pandas as pd
import os
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime
dill._dill._reverse_typemap['ClassType'] = type

# Создание объекта Flask
app = flask.Flask(__name__)

# Инициализация модели как None
model = None

# Настройка логгера для записи в файл
handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# Функция для загрузки модели из файла
def load_model(model_path):
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


# Путь к файлу с моделью
modelpath = "/app/app/models/logreg_pipeline.dill"
load_model(modelpath)


# Маршрут для базовой страницы
@app.route("/", methods=["GET"])
def general():
    return """Добро пожаловать на страницу машинного обучения."""


# Маршрут для предсказания на основе входных данных
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")

    # Проверка, были ли данные отправлены методом POST
    if flask.request.method == "POST":
        message = ""
        request_json = flask.request.get_json()

        # Проверка наличия ключа "message" в JSON-запросе
        if request_json["message"]:
            message = request_json['message']

        # Запись информации в лог
        logger.info(f'{dt} Data: description={message}')
        try:
            preds = model.predict_proba(pd.DataFrame({"message": [message]}))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        data["success"] = True

    return flask.jsonify(data)


# Загрузка модели и запуск сервера Flask
if __name__ == "__main__":
    print(("* Загрузка модели и запуск сервера Flask..."
           "пожалуйста, подождите, пока сервер полностью запустится"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
