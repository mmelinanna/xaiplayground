import numpy as np
import pandas as pd


# -------------------------------------------- DATEN --------------------------------------------------

# Erstellen einer Beispiel-Zeitreihe
dates = pd.date_range('2023-01-01', periods=100)
values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

# DataFrame erstellen
data = pd.DataFrame({'date': dates, 'value': values})


import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Daten skalieren
scaler = MinMaxScaler()
data['value_scaled'] = scaler.fit_transform(data['value'].values.reshape(-1, 1))

# Funktion zur Erstellung von Sequenzen
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sequenzen erstellen
SEQ_LENGTH = 10
X, y = create_sequences(data['value_scaled'].values, SEQ_LENGTH)

# CNN-Modell erstellen
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(SEQ_LENGTH, 1)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=8, validation_split=0.1)

# Vorhersagen erstellen
X_pred = data['value_scaled'].values[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
predicted = model.predict(X_pred)
predicted = scaler.inverse_transform(predicted).flatten()


from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from bokeh.models.widgets import Div
from bokeh.io import output_notebook

# Aktiviert die Ausgabe im Jupyter Notebook (falls verwendet)
output_notebook()

# Original- und Vorhersagedaten kombinieren
predicted_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=len(predicted))
predicted_data = pd.DataFrame({'date': predicted_dates, 'value': predicted})

combined_data = pd.concat([data, predicted_data])

# Datenquelle für Bokeh
source_original = ColumnDataSource(data=dict(date=data['date'], value=data['value']))
source_predicted = ColumnDataSource(data=dict(date=predicted_data['date'], value=predicted_data['value']))

# Bokeh-Plot erstellen
p = figure(x_axis_type='datetime', title="Zeitreihe mit CNN-Vorhersagen", plot_height=400, plot_width=800)
p.line('date', 'value', source=source_original, line_width=2, color='navy', legend_label='Original')
p.line('date', 'value', source=source_predicted, line_width=2, color='orange', legend_label='Vorhersagen')

# Bild-URL relativ zum Projektverzeichnis
url = "static/image1.png"

# Div-Element mit Bild
image_div = Div(text=f"""<img src="{url}" alt="Image" width="400">""")

# Weitere Elemente hinzufügen (Beispiel)
text_div = Div(text="<h2>Hier ist ein Bild:</h2>")

# Column mit den Div-Elementen erstellen
layout = column(p, text_div, image_div)

# Layout anzeigen
output_file("time_series_with_cnn.html")
show(layout)
