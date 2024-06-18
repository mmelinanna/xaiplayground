from bokeh.io import curdoc, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Dropdown
from bokeh.plotting import figure

# Beispiel-Datenquellen
data_item_1 = {'x': [1, 2, 3, 4, 5], 'y': [6, 7, 2, 4, 5]}
data_item_2 = {'x': [1, 2, 3, 4, 5], 'y': [3, 8, 5, 2, 4]}
data_item_3 = {'x': [1, 2, 3, 4, 5], 'y': [5, 3, 6, 7, 2]}

# Erstelle eine ColumnDataSource
source = ColumnDataSource(data=data_item_1)

# Erstelle einen Plot
plot = figure(title="Dropdown-Linked Plot", x_axis_label='X-Achse', y_axis_label='Y-Achse')
plot.line('x', 'y', source=source, line_width=2)

# Dropdown-Menü
menu = [("Item 1", "item_1"), ("Item 2", "item_2"), ("Item 3", "item_3")]

dropdown = Dropdown(label="Dataset", button_type="warning", menu=menu)

# CustomJS Callback zur Aktualisierung des Plots
callback = CustomJS(args=dict(source=source), code="""
    var data_item_1 = {x: [1, 2, 3, 4, 5], y: [6, 7, 2, 4, 5]};
    var data_item_2 = {x: [1, 2, 3, 4, 5], y: [3, 8, 5, 2, 4]};
    var data_item_3 = {x: [1, 2, 3, 4, 5], y: [5, 3, 6, 7, 2]};
    
    if (this.item == 'item_1') {
        source.data = data_item_1;
    } else if (this.item == 'item_2') {
        source.data = data_item_2;
    } else if (this.item == 'item_3') {
        source.data = data_item_3;
    }
    source.change.emit();
""")

dropdown.js_on_event("menu_item_click", callback)

# Layout erstellen und zur Dokumentation hinzufügen
layout = column(dropdown, plot)
curdoc().add_root(layout)
show(layout)
