import streamlit as st
import pandas as pd
import numpy as np
import datetime
from cat_model import CatBoostRegressor
from scaler import RobustScaler

coordinates = pd.read_csv('coordinates.csv')

st.title('Демо-версия сервиса по оценке квартир')
st.markdown('**Это демонстрационный вариант**')
st.markdown('Укажите параметры квартиры и узнайте ее стоимость')

show_data = st.sidebar.checkbox('Все параметры')
st.sidebar.info('Чем больше параметров вы укажете, тем точнее будет прогноз цены.')
st.sidebar.info('Заполните форму **Дополнительные параметры**')

#with st.form('form'):
a = st.text_area('Субъект: ', 'Москва')
b = st.text_area('Общая площадь: ', '50')
c = st.text_area('Количество комнат: ', '2')
if show_data == True:
    st.markdown('## Дополнительные параметры')
    st.subheader('Укажите другие параметры квартиры:')
    d = st.text_area('Этаж: ', '5')
    e = st.text_area('Этажность дома: ', '10')
    f = st.text_area('Площадь кухни: ', '10')
else:
    d = 6
    e = 12
    f = 10

st.sidebar.markdown('Выберите вид жилья')
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


data = {'state':  str(a), 'area': int(b), 'rooms': int(c), 'level': int(d), 'levels': int(e), 'kitchen_area': int(f), 'object_type': int(g), 'building_type': int(h)}
df = pd.DataFrame (data, columns = ['state','area','rooms', 'level', 'levels', 'kitchen_area', 'object_type', 'building_type'], index=[0])

#Добавляем координаты по субъекту
df_with_coordinates = pd.merge(df, coordinates.loc[coordinates.state==a][['geo_lat', 'geo_lon', 'state']], on='state').drop('state', axis=1)

#Добавляем временной признак
now = datetime.datetime.now()
first_date = datetime.datetime(2018, 2, 19)
df_with_coordinates['day_delta'] = (now - first_date).days

st.write(df_with_coordinates)

nums = df_with_coordinates.drop(['object_type', 'building_type'], axis=1) 
st.write(nums)

#Нормализуем числовые признаки
scaler = RobustScaler()
scaled_nums = pd.DataFrame(scaler.get_scaled_data(nums))
#df_scaled_nums = pd.DataFrame(scaled_nums)
st.write(scaled_nums)
ready_df = pd.concat([scaled_nums, df_with_coordinates['object_type'], df_with_coordinates['building_type']], axis=1)
#ready_df = pd.concat([df_scaled_nums, df_with_coordinates[cat_features]], axis=1)

#model = LightGBM()
model = CatBoostRegressor()
prediction = model.predict_price(ready_df)

if st.button('Узнать рекомендованную стоимость'):
    st.markdown('**Рекомендованная цена квартиры**')
    st.subheader(np.round(prediction[0]))
else:
    st.write('Нажмите на кнопку, чтобы рассчитать стоимость!')


