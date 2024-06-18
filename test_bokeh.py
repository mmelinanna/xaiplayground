from bokeh.io import curdoc, show
from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.models import ColumnDataSource, CustomJS, Dropdown
from bokeh.plotting import figure
import numpy as np

##------------ XAI Bild-----------------------------------------------------------------

# Bild-URL
url = "static/XAI_logo.png"
#/Users/melina/github_xaiplayground/xaiplayground/static/XAI_logo.png

# Div-Element mit Bild
image_div = Div(text=f"""<img src="{url}" alt="Image" width="400">""")


##------------ SELECT DATASET -----------------------------------------------------------------

# Beispiel-Daten
data_item_1 = {'x': np.arange(1, 101), 'y': np.random.randn(100) * 10 + 50}
data_item_2 = {'x': np.arange(1, 101), 'y': np.random.randn(100) * 10 + 100}
data_item_3 = {'x': np.arange(1, 101), 'y': np.random.randn(100) * 10 + 150}

# Initiale Datenquelle
source = ColumnDataSource(data=data_item_1)

# Erstelle einen Plot
plot = figure(title="Dropdown-Linked Plot", x_axis_label='X-Achse', y_axis_label='Y-Achse')
plot.line('x', 'y', source=source, line_width=2)

# Dropdown-Menü
menu_dataset = [("Dataset 1", "data1"), ("Dataset 2", "data2"), ("Dataset 3", "data3")]

dropdown_dataset = Dropdown(label="Select Dataset", button_type="warning", menu=menu_dataset)

# CustomJS Callback zur Aktualisierung des Plots
callback = CustomJS(args=dict(source=source, plot = plot), code="""
    var data1 = {x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y: [50, 60, 55, 52, 58, 62, 68, 65, 70, 72]};
    var data2 = {x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y: [100, 105, 98, 102, 108, 112, 115, 118, 120, 125]};
    var data3 = {x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y: [150, 155, 158, 160, 165, 170, 175, 180, 185, 190]};
    
    if (this.item == 'data1') {
        plot.title.text = "Dataset 1";            
        source.data = data1;
    } else if (this.item == 'data2') {
    plot.title.text = "Dataset 2"; 
        source.data = data2;
    } else if (this.item == 'data3') {
        plot.title.text = "Dataset 3" ;
        source.data = data3;
    }
    source.change.emit();
""")

dropdown_dataset.js_on_event("menu_item_click", callback)

##------------ SELECT MODEL -----------------------------------------------------------------

# Beispiel-Daten
source = ColumnDataSource(data=dict(x=[1, 2, 3], y=[4, 5, 6]))

# Erstellen einer Figur
p = figure(title="Select Model from Dropdown", x_axis_label='x', y_axis_label='y')
p.line('x', 'y', source=source, line_width=2)

# Dropdown-Menü
menu_model = [("Linear Regression", "linear_regression"), ("XGBoost", "xgboost"), ("Random Forest Regressor", "random_forest")]

dropdown_model = Dropdown(label="Select Model", button_type="warning", menu=menu_model)

# Callback-Funktion
callback = CustomJS(args=dict(p=p), code="""
    var selected = cb_obj.item;
    if (selected == "linear_regression") {
        p.title.text = "Linear Regression Selected";
    } else if (selected == "xgboost") {
        p.title.text = "XGBoost Selected";
    } else if (selected == "random_forest") {
        p.title.text = "Random Forest Regressor Selected";
    }
""")

dropdown_model.js_on_event('menu_item_click', callback)

##------------ MULTI SELECT -----------------------------------------------------------------


from bokeh.io import show
from bokeh.models import MultiChoice, Tooltip

OPTIONS = ["apple", "mango", "banana", "tomato"]

tooltip = Tooltip(content="Choose any number of the items", position="right")

multi_choice = MultiChoice(value=OPTIONS[:2], options=OPTIONS, title="Choose values:", description=tooltip)

#show(multi_choice)

##------------ DataRangePicker -----------------------------------------------------------------

from bokeh.io import show
from bokeh.models import CustomJS, DateRangePicker

date_range_picker = DateRangePicker(
    title="Select date range",
    value=("2019-09-20", "2019-10-15"),
    min_date="2019-08-01",
    max_date="2019-10-30",
    width=400,
)
date_range_picker.js_on_change("value", CustomJS(code="""
    console.log("date_range_picker: value=" + this.value, this.toString())
"""))

#show(date_range_picker)

# Layout erstellen und zur Dokumentation hinzufügen
layout_sidebar = column(image_div, dropdown_dataset, dropdown_model, multi_choice, date_range_picker)
overall = row(layout_sidebar, plot, p)
curdoc().add_root(overall)


