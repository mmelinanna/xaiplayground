""" IN DEVELOPMENT"""
import pandas as pd
from bokeh.models import ColumnDataSource
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA


class model_creator(object):
    def __init__(self, d_source, split_ind, model_selection="SARIMAX", conf_i=0.95):
        self.d_source = d_source
        self.split_ind = split_ind
        self.model_selection= model_selection
        self.conf_i = conf_i

        #Create Train Test Split
        train_df = pd.DataFrame(data={"time": d_source.data["time"][0:split_ind], "values": d_source.data["synthetic_data"][0:split_ind]})   
        test_df = pd.DataFrame(data={"time": d_source.data["time"][split_ind-1:], "values": d_source.data["synthetic_data"][split_ind-1:]})

        #Select Model + Fit
        def create_model(model_selection, conf_i=0.95):
            if self.model_selection == "SARIMAX":
                current_model_ = SARIMAX(train_df["values"], order = (1,1,1), seasonal_order=(1,1,0,12), enforce_stationarity=False, enforce_invertibility=False)
                current_model = current_model_.fit()
                print(current_model.summary().tables[1])
                y_pred = current_model.get_forecast(len(test_df.index))
                y_pred_df = y_pred.conf_int(alpha = 1-conf_i) 
                y_pred_df["Predictions"] = current_model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
                y_pred_df.index = test_df["time"]
                y_pred_out = y_pred_df["Predictions"] 
                current_model.plot_diagnostics(figsize=(16, 8))


            elif self.model_selection == "RF":
                pass
    
        self.current_model = model_selection

        #ColumnDataSource Generation
        def initiale_source(y_pred_df, tr):
            y_pred_CDS = ColumnDataSource({"time": y_pred_df.index,
                                    "lower_y": y_pred_df.loc[:,"lower values"].values,
                                    "upper_y": y_pred_df.loc[:,"upper values"].values,
                                    "predictions": y_pred_df.loc[:,"Predictions"].values})
            y_train_CDS = ColumnDataSource({"time": train_df["time"],
                                        "value": train_df["values"]})
            y_test_CDS = ColumnDataSource({"time": test_df["time"],
                                        "value": test_df["values"]})
            
            self.y_pred_CDS = y_pred_CDS
            self.y_train_CDS = y_train_CDS
            self.y_test_CDS = y_test_CDS

        def stream_source():
            self.y_pred_CDS.data = ""

