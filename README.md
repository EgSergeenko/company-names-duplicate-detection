# Обнаружение дубликатов среди названий компаний

### Запуск
1. Используя файл можно установить зависимости requirements.txt:

   <code>python -m pip install -r requirements.txt</code>

2. Для запуска обучения:

   <code>python src/train.py</code>
3. Для запуска инференса:
   
   <code>python src/test.py +threshold=<i>float</i></code>

### Структура репозитория
* <code>configs</code> - конфигурационные файлы с гиперпараметрами
* <code>data/raw</code> - исходный датасет
* <code>data/processed</code> - предобработанный датасет
* <code>docs</code> - картинки и прочее
* <code>models</code> - веса моделей
* <code>notebooks</code> - jupyter ноутбуки
* <code>setup.cfg</code> - настройки flake8, mypy и т. д.
* <code>src</code> - папка с исходным кодом

### Работа с данными
Исходные данные представляли из себя 4 колонки 
1. ***pair_id*** - номер пары
2. ***name_1*** - название компании 1
3. ***name_2*** - название компании 2
4. ***is_duplicate*** - является ли пара названий дубликатом 

Пример исходных данных:

| pair_id | name_1 | name_2 | is_duplicate |
| :-----: | :----: |:-----: | :----------: |
| 1 | Iko Industries Ltd. | Enormous Industrial Trade Pvt., Ltd. | 0 |
| 2 | Apcotex Industries Ltd. | Technocraft Industries (India) Ltd. | 0 |
| 381870 | iko sales | IKO SLOVAKIA | 1 |

### Модель
Из-за того что в данной задаче множество классов объектов не является конечным, для решения был выбран подход metric learning.
Для построения embedding'ов названий компаний была использована следующая архитектура:
```python
Model(
  (lstm): LSTM(<symbols_number>, <hidden_size>, num_layers=2, dropout=<dropout_fraction>, bidirectional=True)
  (fc1): Linear(in_features=<hidden_size> * 2, out_features=<embedding_size> * 2, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=<dropout_fraction>)
  (fc2): Linear(in_features=<embedding_size>, out_features=<embedding_size> * 2, bias=True)
)
```
На вход модели подавались символы названия, преобразованные с помощью one-hot encoding.
В итоге для каждого названия был получен тензор размера <i>(L, S)</i>,
где <i>L</i> - количество символов в названии, <i>S</i> - количество символов в алфавите.

Полученные с помощью модели embedding'и были в дальнейшем использованы для нахождения дубликатов.
В множестве объектов выделялась группа <i>anchors</i>, в которую входили названия, дубликаты которых необходимо обнаруживать.
Вторая группа — <i>queries</i> — объекты, которые соотносились с <i>anchors</i>.
Для каждого названия из <i>queries</i> рассчитывалось расстояние до каждого объекта из <i>anchors</i>.
Если какое-то из расстояний было меньше заданного порогового значения, то данный <i>query</i> относился к классу '1' (дубликаты).
В противном случае название получало метку '0' (не дубликат).

Выбранная модель обучалась с использованием ArcFaceLoss в качестве функции потерь.

Батчи, передаваемые в модель, формировались таким образом, чтобы для каждого объекта была пара с такой же меткой класса.

### Валидация
Данные были разбиты на три множества (<i>train</i>=0.9, <i>val</i>=0.05, <i>test</i>=0.05) с непересекающимися метками кассов.
Часть <i>train</i> использовалась для обучения модели, на ней рассчитывалось значение функции потерь.
На множестве <i>val</i> подбиралось пороговое значение, на основании которого объект получал метку либо '0', либо '1'.
Лучшее значение использовалось для расчета метрик на части <i>test</i>.

Каждая из частей <i>val</i> и <i>test</i> делилась на <i>queries</i> и <i>anchors</i> следующим образом:
в <i>anchors</i> добавлялось по одному объекту каждого класса, мощность которых >= 2,
все остальные объекты помещались в <i>queries</i>.

Для валидации и тестирования не использовались сгененрированные названия.


### Результаты
Для оценки качества работы модели была выбрана метрика F1, так как она позволяет балансированить
recall (не пропускать дубликаты) и precision (не относить все объекты к дубликатам).

| Model      | Embedding size  | F1 val     | F1 Test      |
| :--------: | :-------------: | :--------: | :----------: |
| LSTM       | 256             | 0.14540    | 0.06695      |
