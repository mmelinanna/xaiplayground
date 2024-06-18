import numpy as np
import pandas as pd
from bokeh.io import curdoc, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Dropdown
from bokeh.plotting import figure
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Beispiel-Daten
np.random.seed(0)
X = np.arange(1, 101).reshape(-1, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 10

# Trainingsmodelle
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=10, random_state=0),
    "XGBoost": XGBRegressor(n_estimators=10, random_state=0)
}

# Trainingsfunktion
def train_model(model_name):
    model = models[model_name]
    model.fit(X, y)
    y_pred = model.predict(X)
    return {'x': X.squeeze(), 'y': y_pred}

# Initiale Modellvorhersage
initial_model_name = "Linear Regression"
initial_data = train_model(initial_model_name)

# Datenquelle
source = ColumnDataSource(data=initial_data)

# Erstelle einen Plot
plot = figure(title="Model Prediction", x_axis_label='X-Achse', y_axis_label='Y-Achse')
plot.line('x', 'y', source=source, line_width=2, legend_label=initial_model_name)
plot.legend.location = "top_left"

# Dropdown-Menü für Modelle
model_menu = [("Linear Regression", "Linear Regression"), 
              ("Random Forest", "Random Forest"), 
              ("XGBoost", "XGBoost")]

model_dropdown = Dropdown(label="Select Model", button_type="warning", menu=model_menu)

# CustomJS Callback zur Aktualisierung des Plots
callback_code = """
    const model_data = {
        'Linear Regression': {x: %s, y: %s},
        'Random Forest': {x: %s, y: %s},
        'XGBoost': {x: %s, y: %s},
    };
    const data = model_data[this.item];
    source.data = data;
    plot.title.text = 'Model Prediction (' + this.item + ')';
    plot.legend.items = [{label: this.item, renderers: [line]}];
    source.change.emit();
""" % (
    list(X.squeeze()), list(train_model("Linear Regression")['y']),
    list(X.squeeze()), list(train_model("Random Forest")['y']),
    list(X.squeeze()), list(train_model("XGBoost")['y'])
)

callback = CustomJS(args=dict(source=source, plot=plot), code=callback_code)

model_dropdown.js_on_event("menu_item_click", callback)

# Layout erstellen und zur Dokumentation hinzufügen
layout = column(model_dropdown, plot)
curdoc().add_root(layout)
show(layout)
