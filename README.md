# XAIplayground

## SETUP Requirements
- Ensure to have all required libraries, packages and dependencies installed in your current environment. 
- Run the most recent main file (Synthetic_data_app.py) with the highest index via bokeh serve app.py from the command line. (The current index is: 003)
- Navigate to your application folder where your main file is located and run :
  - **bokeh serve --show Synth_data_app_003.py** or 
  - **bokeh serve --dev Synth_data_app_003.py** 

  If you choose the first option there will automatically open a window in your default browser with your bokeh app running. The latter one is for running in development mode. You can access the running app manualy via localhost.

## LIMITATIONS of XAI
>[!CAUTION] 
> Please be aware that the **SARIMAX** model and the **1D-CNN** do currently **not support** explainable approaches through Shapley Calculations. If you choose one or both of these models, the explainability BLOCK will not update. There wont be any shapley-value based plots in your browser. 



## Introduction
This project aims to develop a web-based  user friendly AI playground application where it is able to interact playfully with the underlying data, models and explainability concepts and receive immediate response about the impact of their choices. The central focus is to develop more traceability, transparancy and trust by actively including the end user in the decision making process of the machine learning models and visualize whole process. This is done by following the key concepts: 

## Features
1. **Synthesize Data**: Users can playfully create and change the data by adjusting initialization parameters like the trend, seasonality, amplitude, noise and other things of a time series
2. **Explore Data**: Users can look at the data and reveive the most meaningful insights through numerical and visual features (mean, variance, autocorelation, data distribution, sophisticated visualizations)
3. **model selection**: Users can choose between different models which are trained by the customized data sets
4. **explainability approach**: Users receive information about the decision making process of the current model through various visualizations and explainability concepts like shap, lime etc.
5. **Comparison**: Users can playfully compare two datasets and/or models side by side in order to easiely recognize differences based on theirs choices.

## Technologies Used
- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Python 
- **Backend**:
  - JavaScript (?)
  - Python (bokeh server powered by dash/plotly) 

## Bokeh Server file structure
<pre>
-- myapp(directory)\
   |\
   +---__init__.py\
   +---app_hooks.py\
   +---main.py\
   +---request_handler.py\
   +---static\
   |    +---css\
   |    |   +--- special.css\
   |    |\
   |    +---images\
   |    |   +--- example.png\
   |    |\
   |    +---js\
   |        +---special.js\
   | \
   +---theme.yaml\
   +---templates\
        +---index.html\
</pre>


## SARIMAX Model

**order (Non-Seasonal Order)**:

The order parameter specifies the orders of the non-seasonal components of the SARIMA model.
It is denoted as (p, d, q), where:
p represents the order of the AutoRegressive (AR) component, which captures the linear relationship between the current observation and its past values.
d represents the order of differencing required to make the time series stationary. It indicates how many differences you need to take to achieve stationarity.
q represents the order of the Moving Average (MA) component, which captures the linear relationship between the current observation and past white noise (residuals) terms.

**seasonal_order (Seasonal Order)**:

The seasonal_order parameter specifies the orders of the seasonal components of the SARIMA model.
It is denoted as (P, D, Q, s), where:
P represents the seasonal order of the Seasonal AutoRegressive (SAR) component, which captures the seasonal linear relationship between the current observation and its past values separated by a seasonal period s.
D represents the seasonal order of differencing for seasonal stationarity.
Q represents the seasonal order of the Seasonal Moving Average (SMA) component, which captures the seasonal linear relationship between the current observation and past white noise (residuals) separated by a seasonal period s.


