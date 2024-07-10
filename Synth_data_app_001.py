import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

from bokeh.io import curdoc, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, DateRangeSlider, HelpButton, Tooltip, DataTable
from bokeh.models import NumberFormatter, TableColumn, RadioGroup, Button, CustomJS, SetValue, Div, ColorPicker, Spacer
from bokeh.plotting import figure
from bokeh.models.dom import HTML
from bokeh.themes import Theme
from bokeh.palettes import Spectral6
from bokeh.settings import settings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


#CONSTANTS
MAX_WIDTH_SLIDERS=600
BACKGROUND_C= "#fff"
BORDER_C= "#F7F0DE"   #"white" #   "#F0F0AA"
CONFIG_ROW_C = "#FFF"
BLUE_C= "#125779"
LIGHT_BROWN = "#B7AB87"
LIGHT_GREY="#BFB8A7"
DATASET_LENGTH=90
MAIN_FIG_HEIGHT=340
MIN_WIDTH=400
MODEL_OPTIONS=["SARIMAX", "RF_REGR", "1D-CNN", "XGBOOST"]

settings.default_server_port = 5007

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
              tools="pan,reset,save, xwheel_zoom", margin=(0, 0, -1, 0), background_fill_color=BACKGROUND_C, min_border=30,
              border_fill_color=BORDER_C, styles={"padding-righ":"300", "border-right":"300px"}, min_border_right=60,
              min_border_left=70, x_range=[0, 89], y_range=[-7, 25], align="center")


source = ColumnDataSource(data=dict(time=synthetic_data.index, synthetic_data=synthetic_data.values))
synth_data_line=plot.line('time', 'synthetic_data', source=source, line_width=2.5, line_alpha=0.8,  legend_label="synthetic_data")
plot.legend.location = "top_left"
plot.legend.background_fill_alpha = 0.8
plot.xaxis.axis_label = "time"
plot.yaxis.axis_label = "value"
plot.toolbar.active_scroll = "auto"


text = TextInput(title="title", value='Synthetic Time Series fancy')
slope = Slider(title="slope", value=0.2, start=-1.0, end=1.3, step=0.05, align="center")
amplitude = Slider(title="amplitude", value=2.0, start=-5.0, end=7.0, step=0.25, align="center")
phase = Slider(title="phase", value=0.0, start=0.0, end=2, step=0.5, align="center")
freq = Slider(title="frequency", value=0.1, start=0.02, end=0.3, step=0.02, align="center")
noise = Slider(title="noise", value=0.5, start=0.0, end=2.5, step=0.05, align="center")
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.05, align="center")
date_range_slider = DateRangeSlider(value=(date(2018, 9, 15), date(2020, 9, 15)), step=10, start=date(2016, 9, 1),
                                    end=date(2022, 4, 15), margin=(0, 30, 0, 30), align="center")
train_test_split_slider = Slider(title="train_test_split", value=60, start=45, end=80, step=1, align="center")
line_thickness = Slider(title="line_thickness", value=2.5, start=1.0, end=3.5, step=0.05, align="center")
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
              tools="pan,reset,save,xwheel_zoom", margin=(-1, 0, 0, 0), background_fill_color=BACKGROUND_C, border_fill_color=BORDER_C,
               min_border_left=70, min_border_right=60, x_range=plot.x_range, y_range=[-7, 25], align="center")
train_data_glyph=plot_2.line('time', 'value', source=y_train_CDS, line_width=2.5, line_alpha=0.8, legend_label="train_data")
test_data_glyph=plot_2.line('time', 'value', source=y_test_CDS, line_width=2.5, line_alpha=0.8, line_color="#2ca02c", legend_label="test_data")
pred_data_glyph=plot_2.line('time', 'predictions', source=y_pred_CDS, line_width=2.5, line_alpha=0.9, line_color="#ff7f0e", line_dash="dashdot", legend_label="model_prediction")
plot_2.legend.location = "top_left"
plot_2.legend.background_fill_alpha = 0.8
plot_2.xaxis.axis_label = "time"
plot_2.yaxis.axis_label = "value"
plot_2.toolbar.active_scroll = "auto"


# Include own modules and functions
from dataframe_lagger import time_series_lagger, train_test_split, walk_forward_validation_historic


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
        print(y_pred_df)
        y_pred_out = y_pred_df["Predictions"] 
        #current_model.plot_diagnostics(figsize=(16, 8))

    elif model_selection =="RF_REGR":
        data_concatenated = pd.concat([train_df, test_df.iloc[1:,:]], ignore_index=True)
        tsl = time_series_lagger(data_concatenated.loc[:,"values"].to_list(), n_in=6, n_out=1, dropnan=True)
        print(tsl)
        current_model, mae, y, y_pred = walk_forward_validation_historic(tsl, test_df.shape[0], model_selection="RF")
        y_pred_df = pd.DataFrame(data={"Predictions":y_pred, "lower values":y_pred, "upper values":y_pred})
        y_pred_df.index = test_df["time"]
        print(y_pred_df)
        ##TODO very much --> not so much anymore


    elif model_selection=="1D-CNN":
        data_concatenated = pd.concat([train_df, test_df.iloc[1:,:]], ignore_index=True)
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(data_concatenated)
        tsl = time_series_lagger(data_concatenated.loc[:,"values"].to_list(), n_in=6, n_out=1, dropnan=True)
        #tsl = time_series_lagger(values_scaled, n_in=6, n_out=1, dropnan=True)
        print("tsl_cnn:")
        print(tsl)
        current_model, mae, y, y_pred = walk_forward_validation_historic(tsl, test_df.shape[0], model_selection="CNN")
        y_pred_df = pd.DataFrame(data={"Predictions":y_pred, "lower values":y_pred, "upper values":y_pred})
        y_pred_df.index = test_df["time"]
        print("Mean_Absolute_Error_CNN: "+ str(mae))


    elif model_selection=="XGBOOST":
        data_concatenated = pd.concat([train_df, test_df.iloc[1:,:]], ignore_index=True)
        tsl = time_series_lagger(data_concatenated.loc[:,"values"].to_list(), n_in=6, n_out=1, dropnan=True)
        print(tsl)
        current_model, mae, y, y_pred = walk_forward_validation_historic(tsl, test_df.shape[0], model_selection="XGB")
        y_pred_df = pd.DataFrame(data={"Predictions":y_pred, "lower values":y_pred, "upper values":y_pred})
        y_pred_df.index = test_df["time"]
        print(y_pred_df)


    return current_model, y_pred_df


def update_model_(split_ind=60, model_selection="SARIMAX"):
    model_selection_index = radio_group_models.active
    model_selection = model_selection_dict[model_selection_index]
    print(model_selection)
    if train_test_split_slider.value is not None:
        split_ind=train_test_split_slider.value
    train_df = pd.DataFrame(data={"time": source.data["time"][0:split_ind+1], "values": source.data["synthetic_data"][0:split_ind+1]})   
    test_df = pd.DataFrame(data={"time": source.data["time"][split_ind:], "values": source.data["synthetic_data"][split_ind:]})
    
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


# Read the custom CSS from an external file
with open('custom_styles.css', 'r') as css_file:
    custom_css = css_file.read()

# Create a Div to hold the custom CSS
#css_div = Div(text=custom_css)
css_div = Div(text=f"<style>{custom_css}</style>")


C = Div(text="C", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 40, 0, 0))
#O = Div(text=f"{custom_css}<h1>O</h1>", width=40)
O = Div(text="O", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 40, 0, 0))
N = Div(text="N", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 60, 0,0))
F = Div(text="F", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 40, 0, 0))
I = Div(text="I", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 40, 0, 4))
G = Div(text="G", width=40, styles={'font-size': "1.5rem","font-weight":"400"},margin=(0, 40, 0, 0))
model1_text = Div(text="M1", styles={"font-size": "2rem", "font-weight":"300"}, align="center")
model2_text = Div(text="M2", styles={"font-size": "2rem", "font-weight":"300"}, align="center")



# Load the custom template
with open('custom_template.html', 'r') as file:
    template = file.read()




# -----------------------------------------------DATA TABLE IMPLEMENTATION--------------------------------------------#
columns = [
        TableColumn(field="time", title="time"),
        TableColumn(field="synthetic_data", title="value", formatter=NumberFormatter(format="0.0000")),
    ]

data_table = DataTable(source=source, columns=columns, height=250, editable=True, align="center", margin=(0,40,0,65), sizing_mode="stretch_width")


# -----------------------------------------------CURRENT_DOC REFRESHMENT--------------------------------------------#
button = Button(label="Apply Model", button_type="su", align="center")
button2 = Button(label="Reset Model", button_type="success", align="center")
button2.js_on_event("button_click", SetValue(button, "label", "Apply Model"))

button.on_click(update_model_)
button.js_on_event("button_click", CustomJS(code="console.log('button: click!', this.toString())"))
button.js_on_event("button_click", SetValue(button, "label", "Model applied"))
      
     
for widget_ in [offset,slope, amplitude, phase, freq]:
    widget_.on_change('value', update_data)  
    #widget_.js_on_change("value", SetValue(button, "label", "Apply Model"))  


for line_ in [synth_data_line, train_data_glyph, test_data_glyph, pred_data_glyph]:
    line_thickness.js_link(attr='value', other=line_.glyph, other_attr='line_width')                 

radio_group_models = RadioGroup(labels=MODEL_OPTIONS, active=None, align="center")
radio_group_models2= RadioGroup(labels=MODEL_OPTIONS, active=None, align="center")
model_selection_dict = {key:value for (key,value) in zip(range(4), MODEL_OPTIONS)}

def radio_handler(attrname, old, new):
    print('Radio button option ' + str(new) + ' selected.')

radio_group_models.js_on_event('button_click', CustomJS(code="""
    console.log('radio_group: active=' + this.origin.active, this.toString())
"""))
radio_group_models.on_change("active", radio_handler)
#radio_group_models.js_on_change("change:active", SetValue(button, "label", "Apply XY"))
#radio_group_models.on_event("button_click", radio_handler)

button2.js_on_event("button_click", SetValue(button, "label", radio_group_models.active))

picker = ColorPicker(title="BG_Color_Core")
#picker.js_link('color', line.glyph, 'line_color')


# -----------------------------------------------FINALIZE LAYOUT CURRENT_DOC--------------------------------------------#


# bokeh serve --show Synth_data_app001.py
# bokeh serve Synth_data_app_001.py --dev                        <---DEV-mode
# http://localhost:5006/Synth_data_app

curdoc().title = "Synthetic data"
slope_with_annot= row(slope, help_slope, align="center")
amplitude_with_annot = row(amplitude, help_slope, align="center")
phase_with_annot = row(phase, help_slope, align="center")
freq_with_annot = row(freq, help_slope, align="center")
noise_with_annot = row(noise, help_slope, align="center")

space=Spacer(height=300, sizing_mode="stretch_width")
config_col = column(C, O, N, F, I, G, width=40, align="center", styles={"border-radius": "5px", "margin-right":"40px"})
config_col_2 = column(C, O, N, F, I, G, width=40, align="center", styles={"margin-left":"50px", "margin-right":"-20px"}) 
slider_menu_layout = column(slope_with_annot, amplitude, offset, freq, noise, sizing_mode="stretch_width")
slider_menu_layout_annot = column(slope_with_annot, amplitude_with_annot, phase_with_annot, freq_with_annot,
                                   noise_with_annot, sizing_mode="stretch_width")
button_row = row(button, button2, align="center")
model_first_selection_row = row(model1_text,radio_group_models, styles={"background-color":"rgba(255,127,14,0.8)", "border-radius":"6px"}, align="center") 
model_second_selection_row = row(model2_text, radio_group_models2, styles={"background-color":"#ae1272", "border-radius":"6px"}, align="center")
#model_selection_row= row(model_first_selection_row, model_second_selection_row, align="center")
#model_selection_interface = column(model_selection_row, train_test_split_slider, button_row, line_thickness, sizing_mode="stretch_width", align="center")
model_selection_interface = column(radio_group_models, train_test_split_slider, button_row, line_thickness, sizing_mode="stretch_width", align="center")
core_row_layout = row(config_col, slider_menu_layout, data_table, model_selection_interface, config_col_2 , align="center", margin=0,
                      styles={'font-size': "1.5rem","background-color": "#EEEAE9", "border-radius":"15px", "padding": "10px 10px 10px 10px",
                             # "border-style":"dotted",
                               "border-width":"2.5px"}, sizing_mode="stretch_width")
core_row_layout_border = row(core_row_layout, styles={"background-color": BORDER_C, "padding":"0px 100px 0px 100px"},
                              sizing_mode="stretch_width", align="center")
bottom_row = row(date_range_slider, styles={"background_color": BORDER_C}, align="center") # sizing_mode is incompatible with alignment centered
final_layout = column(plot, core_row_layout_border, plot_2, sizing_mode="stretch_width")

model_selection_interface.styles["background-color"] ="ff7f0e"
cd = curdoc()
#cd.template = template
cd.add_root(final_layout)
cd.theme = Theme(filename="theme.yaml") #improving the modularity of the app and decouple the style layer from the view layer
cd.add_root(css_div)


print(core_row_layout.styles["background-color"])

show(final_layout)




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

### Fine Tuning
#    |--- 1.Interactive Legends (Muting)

#comment 1