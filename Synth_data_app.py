import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, DateRangeSlider
from bokeh.plotting import figure


#CONSTANTS
MAX_WIDTH_SLIDERS=600
BACKGROUND_C= "#fefffa"



'''SYNTHETIC DATA GENERATION 2 (additions are written with CAPS letters)
INCLUDING: {
seasonal_component: amplitude, frequency, SHIFT
trend_component: slope, quadratic_curvature, CUBIC_CURVATURE
noise: standard normal distributed with mean=0, sd=1
AUTOCORELLATION : 
}
'''


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

def generate_synthetic_time_series(t, amplitude, frequency, slope, shift, curvature_quadratic, curvature_cubic,
                                    noise_level, ar_coefficients=None):
    seasonal_component = generate_seasonal_component(t, amplitude, frequency, shift)
    trend_component = generate_trend_component(t, slope, curvature_quadratic, curvature_cubic)
    noise = noise_level * np.random.randn(len(t))
    
    if ar_coefficients is not None:
        ar_data = generate_autoregressive_component(t, ar_coefficients)
    else:
        ar_data = np.zeros_like(t)

    synthetic_data_2 = seasonal_component + trend_component + noise + ar_data
    return synthetic_data_2

# Define parameters    
t = np.linspace(0, 30, 90)
amplitude = 2.0
shift = 0
frequency = 0.1
slope = 0.1
curvature_quadratic = 0.01
curvature_cubic = 0
noise_level = 0.5
ar_coefficients = np.array([0.0, -0.0, 0.0])

synthetic_data = generate_synthetic_time_series(t, amplitude, frequency, slope, shift, curvature_quadratic, curvature_cubic,
                                    noise_level, ar_coefficients=None)
synthetic_data = pd.Series(synthetic_data)


""" MAGIC BOKEH STUFF"""

plot = figure(min_width=400, max_width=1800, height=400, width_policy="max", title="Synthetic time series",
              tools="crosshair,pan,reset,save,wheel_zoom", margin=(0, 40, 10, 40),
              x_range=[0, 90], y_range=[-7, 25], align="center")


source = ColumnDataSource(data=dict(x=synthetic_data.index, y=synthetic_data.values))
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)


#text = TextInput(title="title", value='Synthetic Time Series')
offset = Slider(title="shift", value=0.0, start=-5.0, end=5.1, step=0.1, min_width=200, align="center",
                                max_width=MAX_WIDTH_SLIDERS, width_policy="max",sizing_mode="stretch_width")
slope = Slider(title="slope", value=0.2, start=-1.0, end=1.3, step=0.1, align="center")
amplitude = Slider(title="amplitude", value=2.0, start=-6.0, end=6.0, step=0.5, align="center")
phase = Slider(title="phase", value=0.0, start=0.0, end=2, step=0.5, align="center")
freq = Slider(title="frequency", value=0.1, start=0.02, end=0.3, step=0.02, align="center")
noise = Slider(title="noise", value=0.5, start=0.0, end=2, step=0.1, align="center")
date_range_slider = DateRangeSlider(value=(date(2018, 9, 15), date(2020, 9, 15)), step=10, start=date(2016, 9, 1),
                                    end=date(2022, 4, 15), margin=(0, 30, 0, 30), min_width=200,
                                    max_width=MAX_WIDTH_SLIDERS, width_policy="max",sizing_mode="stretch_width", align="center")


#Components such as row() and column() elements share their sizing mode with all of their children that do not have their own explicit sizing mode.


#def update_title(attrname, old, new):
 #   plot.title.text = text.value

#text.on_change('value', update_title)


def update_data(attrname, old, new):

    # Get the current slider values
    a = amplitude.value
    b = offset.value
    p = phase.value
    k = freq.value
    n = noise.value
    s = slope.value

    synthetic_data_temp = pd.Series(generate_synthetic_time_series(t, a, k, s, p, curvature_quadratic, curvature_cubic,
                                    n, ar_coefficients))
    
    source.data = dict(x=synthetic_data_temp.index, y=synthetic_data_temp.values)


for w in [offset,slope, amplitude, phase, freq, noise]:
    w.on_change('value', update_data)


inputs = column(plot, slope, amplitude, phase, freq, noise)

# bokeh serve --show Synth_data_app.py
curdoc().title = "Synthetic data"
curdoc().add_root(column(plot, slope, amplitude, phase, freq, noise, sizing_mode="stretch_width"))
#curdoc().add_root(inputs, sizing_mode="stretch_width")

##TODO
### ADD length of data ? (Current instances: 90)
### CHANGE to datetime objects ?
### 

