import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, DateRangeSlider, HelpButton, Tooltip, DataTable
from bokeh.models import NumberFormatter, TableColumn, RadioGroup, Button, CustomJS, SetValue
from bokeh.plotting import figure
from bokeh.models.dom import HTML
from bokeh.themes import Theme
from bokeh.palettes import Spectral6

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA


#CONSTANTS
MAX_WIDTH_SLIDERS=600
BACKGROUND_C= "#fefffa"
DATASET_LENGTH=90
MAIN_FIG_HEIGHT=340
MIN_WIDTH=400
MODEL_OPTIONS=["SARIMAX", "RF_regressor", "1D-CNN"]


'''SYNTHETIC DATA GENERATION0.03
INCLUDING: {
seasonal_component: amplitude, frequency, shift
trend_component: slope, quadratic_curvature, CUBIC_CURVATURE
noise: standard normal distributed with mean=0, sd=1
AUTOCORELLATION : 
}
'''


# -----------------------------------------------TIME SERIES GENERATOR FUNDAMENTALS---------------------------------- #

def generate_seasonal_component(t, amplitude, frequency, shift):
    return amplitude * np.sin(2 * np.pi * frequency * t + shift* np.pi)

def generate_trend_component(t, slope, curvature_quadratic, curvature_cubic):
    return slope * t + curvature_quadratic * t**2 + curvature_cubic * t**3

def generate_autoregressive_component(t, ar_coefficients):
    ar_data = np.zeros_like(t)
    for i in range(len(ar_coefficients)):
        if i == 0:
            continue
        ar_data[i] = np.sum(ar_data[:i] * ar_coefficients[:i]) + np.random.randn()    
    return ar_data

def generate_noise(noise_level):
    return np.round(noise_level * np.random.randn(len(t)),4)

def generate_synthetic_time_series(t, amplitude, frequency, slope, shift, offset,
                                   curvature_quadratic, curvature_cubic, ar_coefficients=None):
    seasonal_component = generate_seasonal_component(t, amplitude, frequency, shift)
    trend_component = generate_trend_component(t, slope, curvature_quadratic, curvature_cubic)
    
    if ar_coefficients is not None:
        ar_data = generate_autoregressive_component(t, ar_coefficients)
    else:
        ar_data = np.zeros_like(t)

    synthetic_data_ = offset + seasonal_component + trend_component + ar_data
    return np.round(synthetic_data_, 4)

# SET INITIAL PARAMETERS   
t = np.linspace(0, 30, 90)
amplitude = 2.0               
shift = 0
offset = 0
frequency = 0.1                 
slope = 0.1
curvature_quadratic = 0.0
curvature_cubic = 0
noise_level = 0.5
ar_coefficients = None
#amplitude, or frequency could have irregularities to.

# CREATE INITIAL TIME SERIES
synthetic_data = generate_synthetic_time_series(t, amplitude, frequency, slope, shift, offset, curvature_quadratic,
                                                 curvature_cubic, ar_coefficients=None)
default_noise = generate_noise(noise_level)

synthetic_data = pd.Series(synthetic_data + default_noise)
synthetic_data.round(decimals=4)



#-----------------------------------------------BASIC BOKEH IMPLEMENTATION-------------------------------------------#

plot = figure(min_width=MIN_WIDTH, max_width=1800, height=MAIN_FIG_HEIGHT, width_policy="max", title="Synthetic time series",
              tools="pan,reset,save,wheel_zoom", margin=(0, 40, 10, 40), background_fill_color=BACKGROUND_C,
              x_range=[0, 90], y_range=[-7, 25], align="center")


source = ColumnDataSource(data=dict(time=synthetic_data.index, synthetic_data=synthetic_data.values))
plot.line('time', 'synthetic_data', source=source, line_width=3, line_alpha=0.8,  legend_label="synthetic_data")
plot.legend.location = "top_left"
plot.legend.background_fill_alpha = 0.8
plot.xaxis.axis_label = "time"
plot.yaxis.axis_label = "value"


text = TextInput(title="title", value='Synthetic Time Series fancy')
slope = Slider(title="slope", value=0.2, start=-1.0, end=1.3, step=0.1, align="center")
amplitude = Slider(title="amplitude", value=2.0, start=-6.0, end=6.0, step=0.5, align="center")
phase = Slider(title="phase", value=0.0, start=0.0, end=2, step=0.5, align="center")
freq = Slider(title="frequency", value=0.1, start=0.02, end=0.3, step=0.02, align="center")
noise = Slider(title="noise", value=0.5, start=0.0, end=2, step=0.1, align="center")
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1, align="center")
date_range_slider = DateRangeSlider(value=(date(2018, 9, 15), date(2020, 9, 15)), step=10, start=date(2016, 9, 1),
                                    end=date(2022, 4, 15), margin=(0, 30, 0, 30), min_width=200,
                                    max_width=MAX_WIDTH_SLIDERS, width_policy="max",sizing_mode="stretch_width", align="center")
train_test_split_slider = Slider(title="train_test_split", value=60, start=45, end=80, step=1, align="center")
#Components such as row() and column() elements share their sizing mode with all of their children that do not have their own explicit sizing mode.



# -----------------------------------------------CALLBACK IMPLEMENTATION--------------------------------------------#
def update_title(attrname, old, new):
    plot.title.text = text.value

def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    o_ = offset.value
    p = phase.value
    k = freq.value
    s = slope.value

    synthetic_data_temp = pd.Series(generate_synthetic_time_series(t, a, k, s, p, o_, curvature_quadratic, curvature_cubic,
                                    ar_coefficients)+ default_noise)
    synthetic_data_temp.round(decimals=4)
    source.data = dict(time=synthetic_data_temp.index, synthetic_data=synthetic_data_temp.values)

def update_noise(attrname, old, new):
    global default_noise

    n = noise.value
    updated_noise = generate_noise(n)
    default_noise=updated_noise
    
    a = amplitude.value
    o_ = offset.value
    p = phase.value
    k = freq.value
    s = slope.value

    synthetic_data_temp = pd.Series(generate_synthetic_time_series(t, a, k, s, p, o_, curvature_quadratic, curvature_cubic,
                                    ar_coefficients)+ updated_noise)
    source.data = dict(time=synthetic_data_temp.index, synthetic_data=synthetic_data_temp.values)
    
text.on_change('value', update_title)
noise.on_change("value", update_noise)
for widget_ in [offset,slope, amplitude, phase, freq]:
    widget_.on_change('value', update_data)


# -------------------------------------MODEL IMPLEMENTATION & BOKEH TRANSFORMATION------------------------------------#
# Data Preparation & Train-Test Split

y_train_CDS = ColumnDataSource({"time":[], "value": []})
y_test_CDS = ColumnDataSource({"time":[],"value":[]})
y_pred_CDS = ColumnDataSource({"time": [],"lower_y": [], "upper_y": [],"predictions":[]})

plot_2 = figure(min_width=MIN_WIDTH, max_width=1800, height=MAIN_FIG_HEIGHT, width_policy="max", title="Synthetic time series prediction",
              tools="pan,reset,save,wheel_zoom", margin=(0, 40, 10, 40), background_fill_color=BACKGROUND_C,
                x_range=plot.x_range, y_range=[-7, 25], align="center")
plot_2.line('time', 'value', source=y_train_CDS, line_width=3, line_alpha=0.8, legend_label="Train_data")
plot_2.line('time', 'value', source=y_test_CDS, line_width=3, line_alpha=0.8, line_color="#2ca02c", legend_label="Test_data")
plot_2.line('time', 'predictions', source=y_pred_CDS, line_width=3, line_alpha=0.8, line_color="#ff7f0e", legend_label="model_prediction")
plot_2.legend.location = "top_left"
plot_2.legend.background_fill_alpha = 0.8
plot_2.xaxis.axis_label = "time"
plot_2.yaxis.axis_label = "value"




def create_model(train_df, test_df, model_selection):
    assert model_selection in MODEL_OPTIONS, f"'{model_selection}' is not a valid choise. Please choose from {MODEL_OPTIONS}."

    if model_selection == "SARIMAX":
        ARMAmodel_ = ARIMA(train_df["values"], order = (2, 2, 2))    #ARIMA(p, d, q) -> pdq account for seasonality, trend, and noise in data
        SARIMAXmodel_ = SARIMAX(train_df["values"], order = (1,1,1), seasonal_order=(1,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
        current_model = SARIMAXmodel_.fit()
        print(current_model.summary().tables[1])
        y_pred = current_model.get_forecast(len(test_df.index))
        y_pred_df = y_pred.conf_int(alpha = 0.05) 
        y_pred_df["Predictions"] = current_model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
        y_pred_df.index = test_df["time"]
        y_pred_out = y_pred_df["Predictions"] 
        #current_model.plot_diagnostics(figsize=(16, 8))

    elif current_model =="RF_regressor":
        rf_regressor = RandomForestRegressor(n_estimators=100, max_features="sqrt", max_depth=5)
        current_model = rf_regressor.fit()
        ##TODO very much


    elif current_model=="CNN":
        pass

    return current_model, y_pred_df


def update_model_(split_ind=60, model_selection="SARIMAX"):
    print('button clicked.')
    if train_test_split_slider.value is not None:
        split_ind=train_test_split_slider.value
    train_df = pd.DataFrame(data={"time": source.data["time"][0:split_ind], "values": source.data["synthetic_data"][0:split_ind]})   
    test_df = pd.DataFrame(data={"time": source.data["time"][split_ind-1:], "values": source.data["synthetic_data"][split_ind-1:]})
    
    ####---> GO INTO MODEL FUNC, RETURN:(train_df, test_df, pred_df)
    current_model, pred_df = create_model(train_df=train_df, test_df=test_df, model_selection=model_selection)


    y_train_CDS.data = ({"time": train_df["time"],
                        "value": train_df["values"]})
    y_test_CDS.data = ({"time": test_df["time"],
                        "value": test_df["values"]})
    y_pred_CDS.data = ({"time": pred_df.index,
                        "lower_y": pred_df.loc[:,"lower values"].values,
                        "upper_y": pred_df.loc[:,"upper values"].values,
                        "predictions": pred_df.loc[:,"Predictions"].values})
    

# DEV_STUFF
# split_ind=60
# from dataframe_lagger import time_series_lagger
# train_df = pd.DataFrame(data={"time": source.data["time"][0:split_ind], "values": source.data["synthetic_data"][0:split_ind]})   
# test_df = pd.DataFrame(data={"time": source.data["time"][split_ind-1:], "values": source.data["synthetic_data"][split_ind-1:]})
# current_model, pred_df = create_model(train_df=train_df, test_df=test_df, model_selection="SARIMAX")
# result = time_series_lagger(data=train_df["values"].to_list(), n_in=5, n_out=1, dropnan=True)



# -----------------------------------------------USER EXPLANATION (HTML)----------------------------------------------#
help_slope = HelpButton(tooltip=Tooltip(content=HTML("""
the slope represents the general trend of the time series.<br /> It determines the <b>average increase</b>
in y over a period of time.<br/>More information: <a href="https://en.wikipedia.org/wiki/Slope">slope</a>!
"""), position="right"), align="center")



# -----------------------------------------------DATA TABLE IMPLEMENTATION--------------------------------------------#
columns = [
        TableColumn(field="time", title="time"),
        TableColumn(field="synthetic_data", title="value", formatter=NumberFormatter(format="0.0000")),
    ]

data_table = DataTable(source=source, columns=columns, width=350, height=250, editable=True, align="center", margin=(0,40,0,60))


# -----------------------------------------------CURRENT_DOC REFRESHMENT--------------------------------------------#
button = Button(label="Apply Model", button_type="success", align="center")
button2 = Button(label="Reset Model", button_type="success", align="center")
button2.js_on_event("button_click", SetValue(button, "label", "Apply Model"))

button.on_click(update_model_)
button.js_on_event("button_click", CustomJS(code="console.log('button: click!', this.toString())"))
button.js_on_event("button_click", SetValue(button, "label", "Model applied"))
      
     
for widget_ in [offset,slope, amplitude, phase, freq]:
    widget_.on_change('value', update_data)  
    #widget_.js_on_change("value", SetValue(button, "label", "Apply Model"))  

                   

radio_group_models = RadioGroup(labels=MODEL_OPTIONS, active=None, align="center")

def radio_handler(attrname, old, new):
    print('Radio button option ' + str(new) + ' selected.')

radio_group_models.js_on_event('button_click', CustomJS(code="""
    console.log('radio_group: active=' + this.origin.active, this.toString())
"""))
radio_group_models.on_change("active", radio_handler)
#radio_group_models.js_on_change("change:active", SetValue(button, "label", "Apply XY"))
#radio_group_models.on_event("button_click", radio_handler)

button2.js_on_event("button_click", SetValue(button, "label", radio_group_models.active))


# -----------------------------------------------FINALIZE LAYOUT CURRENT_DOC--------------------------------------------#


# bokeh serve --show Synth_data_app.py
# bokeh serve Synth_data_app.py --dev                        <---DEV-mode
# http://localhost:5006/Synth_data_app

curdoc().title = "Synthetic data"
slope_with_annot= row(slope, help_slope, align="center")
amplitude_with_annot = row(amplitude, help_slope, align="center")
phase_with_annot = row(phase, help_slope, align="center")
freq_with_annot = row(freq, help_slope, align="center")
noise_with_annot = row(noise, help_slope, align="center")

slider_menu_layout = column(slope_with_annot, amplitude, offset, freq, noise, sizing_mode="stretch_width")
slider_menu_layout_annot = column(slope_with_annot, amplitude_with_annot, phase_with_annot, freq_with_annot,
                                   noise_with_annot, sizing_mode="stretch_width")
model_selection_interface = column(radio_group_models, train_test_split_slider, button, button2, sizing_mode="stretch_width", align="center")
core_row_layout = row(slider_menu_layout, data_table, model_selection_interface, align="center")

cd = curdoc()
cd.add_root(column(plot, core_row_layout, plot_2, sizing_mode="stretch_width"))
cd.theme = Theme(filename="theme.yaml") #improving the modularity of the app and decouple the style layer from the view layer








##TODO
### Data Generation
#   |--- ADD length of data ? (Current instances: 90)
#   |--- CHANGE to datetime objects ?
#   |--- Apply Feature: Ground_level/Offset                              √ DONE
#   |--- Apply Feature: Quadratic/Cubic Slope + Corresponding Widget    

### Data Visualization
#    |--- Customizable Autocorrelation Plot (User should choose the lags) (one or span of multiple)
#    |--- distribution of data change

### Forecast Model implementation 
#   |--- 1. State of the Art Deterministic Forecast: (SARMIAX)           √ DONE
#   |--- 2. ML-based: Random Forest                                      - IN PROGRESS 
#   |--- 3. ML-based: XGB Boost                                         
#   |--- 4. Sophisticated ML-model: (CNN), (LSTM)                        - IN PROGRESS

#comment 1