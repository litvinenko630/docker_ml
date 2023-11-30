import pandas as pd
import dill
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Загрузка данных
df = pd.read_csv('../.././data/sms.csv')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df, df['target'], test_size=0.33, stratify=df['target'],
                                                    random_state=42)


# Создание класса для выбора столбца из DataFrame
class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Трансформатор для выбора одного столбца из DataFrame для выполнения дополнительных преобразований
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


# Создание списка трансформаторов
final_transformers = []

# Создание трансформатора для обработки текстовых данных через TF-IDF
message = Pipeline([
    ('selector', ColumnSelector(key='message')),
    ('tfidf', TfidfVectorizer())
])

final_transformers.append(('message', message))

# Объединение всех трансформаторов в FeatureUnion
feats = FeatureUnion(final_transformers)

# Создание пайплайна обработки данных и обучения модели
pipeline = Pipeline([
    ('features', feats),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Обучение модели на обучающих данных
pipeline.fit(X_train, y_train)

# Сохранение обученной модели в файл
with open('./logreg_pipeline.dill', 'wb') as f:
    dill.dump(pipeline, f)

# Получение прогнозов вероятности для тестового набора данных
preds = pipeline.predict_proba(X_test)[:, 1]

# Вычисление precision, recall и thresholds для построения кривой Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, preds)
fscore = (2 * precision * recall) / (precision + recall)
ix = np.argmax(fscore)

# Вывод наилучших метрик для порога классификации
print(f'Лучший порог={thresholds[ix]}, '
      f'F-мера={fscore[ix]:.3f}, '
      f'Precision={precision[ix]:.3f}, '
      f'Recall={recall[ix]:.3f}')

# Загрузка списка стоп-слов на английском языке из библиотеки nltk
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

# Объединение текстов всех сообщений в одну строку
data_text = ",".join(txt.lower() for txt in df['message'])

# Создание облака слов
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      scale=5,
                      stopwords=stop_words,  # Используем загруженные стоп-слова
                      background_color="white").generate(data_text)

# Отображение облака слов и сохранение изображения
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Самые частоупотребляемые слова', fontsize=15)
plt.savefig('../.././image/frequent.png')
plt.show()
