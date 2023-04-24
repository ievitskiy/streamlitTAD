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
        "�������� ��������",
        ["�������� ������ � ������", "������ � ������"]
    )

    if page == "�������� ������ � ������":
        st.title("�������� ������ � ������")
        st.write("�������� �������� �����")

        st.header("�������� ������")
        st.markdown("""����� ������ ��������� � �������� � ������ ��������� �������������� ���� �Vinho Verde�. ��-�� ������� � ������������������� � ���������� �������� ������ ������-���������� (�������) � ����������������� (��������) ���������� (��������, ��� ������ � ������ ���������, ����� ����, ��������� ���� ���� � �.�.).
��� ������ ������ ����� ������������� ��� ������ ������������� ��� ���������. ������ ����������� � �� �������������� (��������, ���������� ��� ������� ������, ��� �������� ��� ������). ��������� ����������� �������� ����� ������������ ��� ����������� ���������� �������� ��� ������ ���.""")


        st.header("�������� ������")
        st.markdown("""��������������� ������:
* fixed acidity � ������������� �����������,
* volatile acidity � ������� �����������,
* citric acid � �������� �������,
* residual sugar � ���������� �����,
* chlorides � �������,
* free sulfur dioxide � ��������� ������� ����,
* total sulfur dioxide � ������� ���� �����,
* density � ���������,
* pH � ���������� ����������,
* sulphates � ��������,
* alcohol � �������� � ����,
* type_white - ����� �� ����,
* type_red - ������� �� ����,
* quality - ��������.
� �������������� ��������� ���������:
* �������� ���� ���� ��������� �������� 1 ��� 0 � ����������� �� ����, � ������ ���� ����������� ����;
* �������� ���� ����������� �� 0 �� 10, ��� 0 ��� ����� �����, � 10 ��� ���������;
� �������� ��������� ���������:
* �������� ���� ���� ��������� �������� 1 ��� 0 � ����������� �� ����, � ������ ���� ����������� ����;
� ������������ ��������� ���������:
* ������������� �����������,
* ������� �����������,
* �������� �������,
* ���������� �����,
* �������,
* ��������� ������� ����,
* ������� ���� �����,
* ���������,
* ���������� ����������,
* ��������,
* �������� � ����""")

    elif page == "������ � ������":
        st.title("������ � ������")
        st.write("�������� �������� �����")
        request = st.selectbox(
            "�������� ������",
            ["RMSE", "������ 5 ������������� ��������", "���������������� ������", "��������"]
        )

        if request == "RMSE":
            st.header("������ �� ������������������ ������")
            y_pred = model.predict(X)
            rmse = sqrt(mean_squared_error(y, y_pred))
            st.write(f"{rmse}")
        elif request == "������ 5 ������������� ��������":
            st.header("������ 5 ������������� ��������")
            first_5_test = test_data.drop(labels=['quality'], axis=1).iloc[:5, :]
            first_5_pred = model.predict(first_5_test)
            for item in first_5_pred:
                st.write(f"{item:.2f}")
        elif request == "���������������� ������":
            st.header("���������������� ������")

            fa = st.number_input("������������� �����������", 4.6, 15.6)

            va = st.number_input("������� �����������", 0.10, 1.33)

            ca = st.number_input("�������� �������", 0.01, 1.00)

            rs = st.number_input("���������� �����", 0.8, 22.0)

            chl = st.number_input("�������", 0.02, 0.60)

            fsd = st.number_input("��������� ������� ����", 1.00, 131.00)

            tsd = st.number_input("������� ���� �����", 8.0, 313.0)

            density = st.number_input("���������", 0.98, 1.00)

            pH = st.number_input("���������� ����������", 2.74, 3.9)

            sulphates = st.number_input("��������", 0.27, 2.0)

            typewine = st.selectbox("��� ����", ['�������', '�����'])
            typered = 0
            typewhite = 1
            if typewine == '�������':
                typered = 1
                typewhite = 0

            alcohol = st.number_input("��������", 8.4, 14.0)

            if st.button('�����������'):
                data = [fa, va, ca, rs, chl, fsd, tsd, density, pH, sulphates, alcohol, typewhite, typered]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)
                st.write(f"������������� ��������: {pred[0]:.2f}")
            else:
                pass

        elif request == "��������":
            st.header("��������")
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
