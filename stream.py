import streamlit as st
import pandas as pd
import numpy as np
from model import LightGBM

coordinates = pd.read_csv('coordinates.csv')

st.title('Демо-версия сервиса по оценке квартир')
st.markdown('**Это демонстрационный вариант**')
st.markdown('Укажите параметры квартиры и узнайте ее стоимость')

show_data = st.sidebar.checkbox('Все параметры')
st.sidebar.info(
    'Чем больше параметров вы укажете, тем точнее будет прогноз цены.')
st.sidebar.info('Заполните форму **Дополнительные параметры**')

with st.form('form'):
    a = st.text_area('Субъект: ', 'Москва')
    b = st.text_area('Общая площадь: ', '50')
    c = st.text_area('Количество комнат: ', '2')
    if show_data == True:
        st.markdown('## Дополнительные параметры')
        st.subheader(
        'Укажите другие параметры квартиры:')
        d = st.text_area('Этаж: ', '5')
        e = st.text_area('Этажность дома: ', '10')
        f = st.text_area('Площадь кухни: ', '10')
    else:
        d = 6
        e = 12
        f = 10
    submit_button = st.form_submit_button('Узнать рекомендованную стоимость')



st.sidebar.markdown('Выберите тип постройки')
select_object_type = st.sidebar.radio('', ('Вторичное жилье', 'Новостройка'))
if select_object_type == 'Вторичное жилье':
    g = 1
if select_object_type == 'Новостройка':
    g = 2

select_building_type = st.sidebar.selectbox('Выберите тип дома', ('Панельный', 'Монолитый', 'Кирпичный', 'Бетонный', 'Деревянный',  'Другое'))
if select_building_type == 'Панельный':
    h = 1
if select_building_type == 'Монолитый':
    h = 2
if select_building_type == 'Кирпичный':
    h = 3
if select_building_type == 'Бетонный':
    h = 4
if select_building_type == 'Деревянный':
    h = 5
if select_building_type == 'Другое':
    h = 0


data = {'state':  str(a), 'General square': int(b), 'Rooms': int(c), 'Level': int(d), 'Levels': int(e), 'Kitchen square': int(f), 'Object_type': int(g), 'Building type': int(h)}
df = pd.DataFrame (data, columns = ['state','General square','Rooms', 'Level', 'Levels', 'Kitchen square', 'Object_type', 'Building type'], index=[0])

#Добавляем координаты по субъекту

df_with_coordinates = pd.merge(df, coordinates.loc[coordinates.state==a][['geo_lat', 'geo_lon', 'state']], on='state').drop('state', axis=1)

model = LightGBM()
prediction = model.predict_price(df_with_coordinates)
st.markdown('**Рекомендованная цена квартиры**')
st.subheader(np.round(prediction[0]))
