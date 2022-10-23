# Обнаружении дубликатов названий компаний

### Запуск
1. Используя файл можно установить зависимости requirements.txt:

    <code>python -m pip install -r requirements.txt</code>

2. 

### Структура репозитория
* <code>configs</code> - конфигурационные файлы с гиперпараметрами
* <code>data/raw</code> - исходный датасет
* <code>data/processed</code> - очищенный датасет
* <code>docs</code> - картинки и прочее
* <code>models</code> - веса моделей
* <code>notebooks</code> - jupiter ноутбуки
* <code>setup.cfg</code> - настройки flake8 и т.д.
* <code>src</code> - папка для всего кода

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


### Результаты
| Модель      | F1 оценка на валидации | 
| :--------: | :-------------: |
| LSTM |      |
