import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Seteamos la direccion de los archivos
folder_path = '/Users/martiniglesias/Downloads/'

# Traemos los archivos con sus nombres
csv_file_names = ['penalty_data.csv', 'GKPenalty.csv', 'penalty_kicks.csv']

# Definimos la funcion que traera los archivos
def load_csv_data(file_path, encoding='utf-8'):
    return pd.read_csv(file_path, encoding=encoding)

# Definimos una función para cargar datos desde una base de datos SQLite
def load_sqlite_data(file_path, query):
    conn = sqlite3.connect(file_path)
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Loopeamos para todos los archivos
csv_data = {}
for csv_file_name in csv_file_names:
    csv_data[csv_file_name] = load_csv_data(os.path.join(folder_path, csv_file_name), encoding='latin1')

#Analisis exploratorio
for csv_file_name, data in csv_data.items():
    print(f"Exploratory Analysis for {csv_file_name}:")
    print(data.head())  # Mostramos las p´rimeras filas
    print(data.describe())  # Un simple análisis para las variables numericas
    print("\n")

# Cargamos el dataset
data = pd.read_csv('/Users/martiniglesias/Downloads/penalty_data.csv',  encoding='latin1')

# Mostramos la estructura del dataset
print(data.info())

# Cargar los datos desde un archivo CSV - cambiar en el caso de que el archivo no corra
df2 = pd.read_csv(r'/Users/martiniglesias/Downloads/penalty_data.csv', encoding= 'windows-1252')

df2

summary = df2.describe()
summary

#conteos
counts_zona = df2['Kick_Direction'].value_counts()
counts_zona

counts_errados = df2['Keeper_Direction'].value_counts()
counts_errados

Pie_conteo=df2['Saved'].value_counts()
Pie_conteo


# Cambiamos los datos de texto por numericos en la columna Foot
pie_pateador = {'R': 0, 'L': 1}
df2['Pie'] = df2['Foot'].map(pie_pateador)

# Cambiamos los datos de texto por numericos en la columna Keeper
arquero_tirar = {'C': 0, 'R': 1, 'L': 2}
df2['Arquero'] = df2['Keeper_Direction'].map(arquero_tirar)

# Cambiamos los datos de texto por numericos en la columna Kick Direction
pateador = {'C': 0, 'R': 1, 'L': 2}
df2['Pateador'] = df2['Kick_Direction'].map(pateador)

# Cambiamos los datos de texto por numericos en la columna Scored
anotado = {'Scored': 1, 'Missed': 0}
df2['Gol'] = df2['Scored'].map(anotado)


# Matriz de correlacion
selected_columns2 = df2[['Pie', 'Arquero', 'Pateador', 'Gol']]
matriz_corr2 = selected_columns2.corr()
matriz_corr2

# Crear y mostrar la matriz de correlación utilizando un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_corr2, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlación')
plt.show()

# Cuadro de lugares de donde se patea
sns.countplot(x=df2['Pateador'])  
plt.title('Pateador')
plt.show()

# Gráfico de dispersión para visualizar la relación las distintas variables más importantes entre si

# Comenzamos comparando pie con arquero
x = 'Pie'
y = 'Arquero'

# Crear el gráfico de dispersión con línea de regresión y coeficiente de correlación
plt.figure(figsize=(8, 6))
ax = sns.regplot(data=df2, x=x, y=y, ci=95, line_kws={'color': 'red'})

# Calcular el coeficiente de correlación de Pearson
correlation_coef = df2[x].corr(df2[y])

# Agregar el coeficiente de correlación al gráfico
ax.text(0.5, 0.9, f'Coeficiente de Correlación: {correlation_coef:.2f}', transform=ax.transAxes, fontsize=12)

# Añadir título y etiquetas de los ejes
plt.title(f'Comparación entre {x} y {y}')
plt.xlabel(x)
plt.ylabel(y)

# Mostrar el gráfico
plt.show()

# Seguimos comparando arquero con la direccion del pateador
x2 = 'Pateador'
y2 = 'Arquero'

# Crear el gráfico de dispersión con línea de regresión y coeficiente de correlación
plt.figure(figsize=(8, 6))
ax2 = sns.regplot(data=df2, x=x2, y=y2, ci=95, line_kws={'color': 'red'})

# Calcular el coeficiente de correlación de Pearson
correlation_coef = df2[x2].corr(df2[y2])

# Agregar el coeficiente de correlación al gráfico
ax2.text(0.5, 0.9, f'Coeficiente de Correlación: {correlation_coef:.2f}', transform=ax2.transAxes, fontsize=12)

# Añadir título y etiquetas de los ejes
plt.title(f'Comparación entre {x2} y {y2}')
plt.xlabel(x2)
plt.ylabel(y2)

# Mostrar el gráfico
plt.show()

# Seguimos comparando el pie con que patea y la direccion del pateador
x3 = 'Pie'
y3 = 'Pateador'

# Crear el gráfico de dispersión con línea de regresión y coeficiente de correlación
plt.figure(figsize=(8, 6))
ax3 = sns.regplot(data=df2, x=x3, y=y3, ci=95, line_kws={'color': 'red'})

# Calcular el coeficiente de correlación de Pearson
correlation_coef = df2[x3].corr(df2[y3])

# Agregar el coeficiente de correlación al gráfico
ax3.text(0.5, 0.9, f'Coeficiente de Correlación: {correlation_coef:.2f}', transform=ax3.transAxes, fontsize=12)


# Añadir título y etiquetas de los ejes
plt.title(f'Comparación entre {x3} y {y3}')
plt.xlabel(x3)
plt.ylabel(y3)

# Mostrar el gráfico
plt.show()

# Seguimos comparando la direccion del pateador y si este fue gol o no
x4 = 'Gol'
y4 = 'Pateador'

# Crear el gráfico de dispersión con línea de regresión y coeficiente de correlación
plt.figure(figsize=(8, 6))
ax4 = sns.regplot(data=df2, x=x4, y=y4, ci=95, line_kws={'color': 'red'})

# Calcular el coeficiente de correlación de Pearson
correlation_coef = df2[x4].corr(df2[y4])

# Agregar el coeficiente de correlación al gráfico
ax4.text(0.5, 0.9, f'Coeficiente de Correlación: {correlation_coef:.2f}', transform=ax4.transAxes, fontsize=12)

# Añadir título y etiquetas de los ejes
plt.title(f'Comparación entre {x4} y {y4}')
plt.xlabel(x4)
plt.ylabel(y4)

# Mostrar el gráfico
plt.show()

# Procesamos los datos
data = data.drop(['No.', 'Match Week', 'Date', 'Player', 'Team', 'Match', 'Time of Penalty Awarded', 'Final Results'], axis=1)
data['Saved'].fillna(0, inplace=True)  # Fill missing values in 'Saved' column

# Codeamos las variables
label_encoder = LabelEncoder()
data['Foot'] = label_encoder.fit_transform(data['Foot'])
data['Kick_Direction'] = label_encoder.fit_transform(data['Kick_Direction'])
data['Keeper_Direction'] = label_encoder.fit_transform(data['Keeper_Direction'])

# Seleccionamos las variables y cual queremos que sea la variable a predecir
X = data[['Foot', 'Kick_Direction', 'Keeper_Direction']]
y = data['Saved']

# Seperamos el dataset para testeo y training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos el modelo. En este caso Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaliamos el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Intentamos predecir en el caso de tener un nuevo penal.
new_scenario = [[label_encoder.transform(['L'])[0], label_encoder.transform(['C'])[0], label_encoder.transform(['R'])[0]]]
chances = model.predict_proba(new_scenario)[:, 1]
print(f"Chances of saving the goal: {chances[0]*100:.2f}%")

# Creamos un nuevo datafram que combine el original con las predicciones
test_predictions = X_test.copy()  # Creamos la variable con los datos obtenidos de x_test
test_predictions['Predicted'] = y_pred  # Añadimos una columna para las predicciones del modelo

# Lo guardamos a todo en un archivo parquet
test_predictions.to_parquet('test_predictions.parquet', index=False)