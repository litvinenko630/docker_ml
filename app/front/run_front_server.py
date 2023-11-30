from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import StringField
from wtforms.validators import DataRequired
import urllib.request
import json


# Определение класса формы для данных клиента
class ClientDataForm(FlaskForm):
    message = StringField('Message', validators=[DataRequired()])


# Создание экземпляра Flask
app = Flask(__name__)

# Настройка параметров приложения
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)


# Функция для получения предсказания
def get_prediction(message):
    # Создание тела запроса с переданным сообщением
    body = {'message': message}

    myurl = "http://0.0.0.0:8180/predict"
    req = urllib.request.Request(myurl)  # Создание запроса к указанному URL

    # Добавление заголовка с указанием типа содержимого как JSON и его кодировки
    req.add_header('Content-Type', 'application/json; charset=utf-8')

    jsondata = json.dumps(body)  # Преобразование словаря в строку JSON
    jsondataasbytes = jsondata.encode('utf-8')  # Преобразование строки JSON в байты (необходимо для отправки)

    # Добавление заголовка с указанием длины данных в байтах
    req.add_header('Content-Length', str(len(jsondataasbytes)))

    # Отправка запроса на указанный URL с данными в формате JSON
    response = urllib.request.urlopen(req, jsondataasbytes)

    # Получение ответа и извлечение предсказания из полученных данных в формате JSON
    return json.loads(response.read())['predictions']


# Маршруты для обработки запросов
# Маршрут для обработки корневой страницы
@app.route("/")
def index():
    return render_template('index.html')


# Маршрут, который принимает параметр <response> из URL и отображает данные на странице predicted.html
@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)  # Преобразование строки JSON в объект Python
    print(response)  # Вывод данных в консоль (для отладки)
    return render_template('predicted.html', response=response)  # Отображение данных на странице predicted.html


# Маршрут для отображения формы и обработки данных из неё
@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()  # Создание экземпляра формы
    data = dict()
    if request.method == 'POST':  # Проверка, были ли данные отправлены методом POST
        data['message'] = request.form.get('message')  # Получение данных из формы

        try:
            response = str(get_prediction(data['message']))  # Получение предсказания на основе данных из формы
            print(response)  # Вывод предсказания в консоль (для отладки)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})  # Обработка ошибки соединения

        return redirect(url_for('predicted', response=response))  # Перенаправление на страницу с предсказанием
    return render_template('form.html', form=form)  # Отображение формы на странице form.html


# Запуск приложения Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
