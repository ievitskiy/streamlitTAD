# -*- coding: cp1251 -*-
import pickle

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
from math import sqrt
import random


def main():
    model = load_model("model_dumps/model.pkl")
    test_data = load_test_data("data/preprocessed_data.csv")
    y = test_data['quality']
    X = test_data.drop(['quality'], axis=1)

    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""Набор данных относится к красному и белому вариантам португальского вина «Vinho Verde». Из-за проблем с конфиденциальностью и логистикой доступны только физико-химические (входные) и органолептические (выходные) переменные (например, нет данных о сортах винограда, марке вина, продажной цене вина и т.д.).
Эти наборы данных можно рассматривать как задачи классификации или регрессии. Классы упорядочены и не сбалансированы (например, нормальных вин намного больше, чем отличных или плохих). Алгоритмы обнаружения выбросов можно использовать для определения нескольких отличных или плохих вин.""")


        st.header("Описание данных")
        st.markdown("""Предоставленные данные:
* fixed acidity – фиксированная кислотность,
* volatile acidity – летучая кислотность,
* citric acid – лимонная кислота,
* residual sugar – остаточный сахар,
* chlorides – хлориды,
* free sulfur dioxide – свободный диоксид серы,
* total sulfur dioxide – диоксид серы общий,
* density – плотность,
* pH – водородный показатель,
* sulphates – сульфаты,
* alcohol – алкоголь в вине,
* type_white - белое ли вино,
* type_red - красное ли вино,
* quality - качество.
К категориальным признакам относятся:
* признаки типа вина принимают значение 1 или 0 в зависимости от того, к какому типу принадлежит вино;
* качество вина оценивается от 0 до 10, где 0 это супер плохо, а 10 это идеальное;
К бинарным признакам относятся:
* признаки типа вина принимают значение 1 или 0 в зависимости от того, к какому типу принадлежит вино;
К вещественным признакам относятся:
* фиксированная кислотность,
* летучая кислотность,
* лимонная кислота,
* остаточный сахар,
* хлориды,
* свободный диоксид серы,
* диоксид серы общий,
* плотность,
* водородный показатель,
* сульфаты,
* алкоголь в вине""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["RMSE", "Первые 5 предсказанных значений", "Пользовательский пример", "Пасхалка"]
        )

        if request == "RMSE":
            st.header("Корень из среднеквадратичной ошибки")
            y_pred = model.predict(X)
            rmse = sqrt(mean_squared_error(y, y_pred))
            st.write(f"{rmse}")
        elif request == "Первые 5 предсказанных значений":
            st.header("Первые 5 предсказанных значений")
            first_5_test = test_data.drop(labels=['quality'], axis=1).iloc[:5, :]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                st.write(f"{item:.2f}")
        elif request == "Пользовательский пример":
            st.header("Пользовательский пример")

            fa = st.number_input("Фиксированная кислотность", 4.6, 15.6)

            va = st.number_input("Летучая кислотность", 0.10, 1.33)

            ca = st.number_input("Лимонная кислота", 0.01, 1.00)

            rs = st.number_input("Остаточный сахар", 0.8, 22.0)

            chl = st.number_input("Хлориды", 0.02, 0.60)

            fsd = st.number_input("Свободный диоксид серы", 1.00, 131.00)

            tsd = st.number_input("Диоксид серы общий", 8.0, 313.0)

            density = st.number_input("Плотность", 0.98, 1.00)

            pH = st.number_input("Водородный показатель", 2.74, 3.9)

            sulphates = st.number_input("Сульфаты", 0.27, 2.0)

            typewine = st.selectbox("Тип вина", ['Красное', 'Белое'])
            typered = 0
            typewhite = 1
            if typewine == 'Красное':
                typered = 1
                typewhite = 0

            alcohol = st.number_input("Алкоголь", 8.4, 14.0)

            if st.button('Предсказать'):
                data = [fa, va, ca, rs, chl, fsd, tsd, density, pH, sulphates, alcohol, typewhite, typered]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"Предсказанное значение: {pred[0]:.2f}")
            else:
                pass

        elif request == "Пасхалка":
            st.header("Пасхалка")
            st.write(":)")


@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";")

    return df


if __name__ == "__main__":
    main()
