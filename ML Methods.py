
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS
from neuralforecast.auto import AutoNHITS
from numpy.ma.core import append
from pytorch_forecasting.metrics import MAE
from neuralprophet import NeuralProphet
import time
from torch.nn import L1Loss
import torch.nn.functional as F
from neuralforecast.losses import MAE

# import data
dataDf = pd.read_csv("C:/Users/cb241/OneDrive/Documents/1 - University/MAppFin/Semester H/FINAN521 - Finance Project/Data/FINAN521 - Dataset - Final.csv")
dataDf['date'] = pd.to_datetime(dataDf['date'], dayfirst=True)
runcount =0
colcount=0
h = 4

numtest = 5#200
minTrainLen = len(dataDf)-numtest-1#375
model_test_metrics = pd.DataFrame(columns=['Run', 'Model_name', 'MAE', 'RMSE'])

for col in dataDf.columns[1:2]: #remove 2 for final
    #set target variable
    target = col
    colcount = colcount+1
    start_time = time.perf_counter()
    xxx = len(dataDf)
    for i in range(minTrainLen, len(dataDf)-h):
        print(i)
        runcount += 1
        trainDf = dataDf[['date', target]].iloc[0:i+1,].rename(columns={'date': 'ds', target: 'y'})
        testDf = dataDf[['date', target]].iloc[i+1:i+1+h,].rename(columns={'date': 'ds', target: 'y'})

        trainDf['unique_id'] = 1
        testDf['unique_id'] = 1
        testDf = testDf.reset_index(drop=True)

    ##################################################################################################
    # models
    ##################################################################################################

    # simple ave model - for checking against R

        train_ave = trainDf['y'].mean()

        simple_ave_predict = pd.DataFrame({
            'date': testDf['ds'],
            'prediction': [train_ave]*len(testDf)
        }
        )

    # nhits model
    # nhits model spec
        nhits_model = NHITS(h = h,
                  input_size = 24,
                  max_steps = 350,
                  learning_rate = 0.001,
                  n_pool_kernel_size= [1, 1, 1],
                            loss=MAE()
                  )

    # nhits fit and forecast
        nhit_fcast = NeuralForecast(models=[nhits_model], freq='MS')
        nhit_fcast.fit(df = trainDf)
        nhits_forecast = nhit_fcast.predict(futr_df=testDf)


    #nbeats model
        #nbeats_model = NBEATS(h = h,
                    #input_size=12, #number of lags
                    #max_steps = 2
                    #n_blocks=[1, 1, 1] #Default used -
                    #mlp_units=[[512, 512], [512, 512]],
                    #)
    # nbeats fit and forecast
        #nbeats_fcast = NeuralForecast(models=[nbeats_model], freq='MS')
        #nbeats_fcast.fit(df=trainDf)
        #nbeats_forecast = nbeats_fcast.predict(futr_df=testDf)


    #neural Prophet
        #neural_lags = 12
        #neural_trainDf = trainDf.iloc[neural_lags:].drop(columns=['unique_id'])
        #test_with_lags = pd.concat([trainDf[-neural_lags:], testDf], ignore_index=True)
        #neural_testDf = test_with_lags.drop(columns=['unique_id'])


        #neural_prophet_model = NeuralProphet(n_lags = neural_lags, n_forecasts= h, changepoints_range=0.9)
        #neural_prophet_fit = neural_prophet_model.fit(neural_trainDf, freq='MS')
        #neural_prophet_forecast = neural_prophet_model.predict(neural_testDf)
        #neural_prophet_forecast_sub = neural_prophet_forecast.tail(h).reset_index(drop=True)

        #cols_select = [col for col in neural_prophet_forecast_sub.columns if col.startswith("yhat")]
        #neural_prophet_forecast_sub["yhat1"] = neural_prophet_forecast_sub[cols_select].bfill(axis=1).iloc[:, 0]
        #neural_prophet_forecast_sub = neural_prophet_forecast_sub[["ds", "y", "yhat1"]]

    ##################################################################################################
    # add model forecasts to array
    ##################################################################################################

    # date, actual and forecasts
        all_model_data = pd.DataFrame({
            'date': testDf.ds,
            'Actual': testDf.y,
            'AVG': simple_ave_predict.prediction,
            'nHits_forecast': nhits_forecast['NHITS'],
            #'nBeats_forecast': nbeats_forecast['NBEATS'],
            #'neural_prophet_forecast': neural_prophet_forecast_sub['yhat1'],
        })

    # find num cols and define list for errors
        numcols = all_model_data.shape[1]
        mae_value_store = []
        rmse_values_store = []

    # loop through models (columns), calculate MAE and RMSE
        for j in range(2, numcols):
            mae_values = sum(abs(all_model_data['Actual'] - all_model_data.iloc[:,j]))/len(all_model_data)
            rmse_values = ((sum((all_model_data['Actual']-all_model_data.iloc[:,j])**2))/len(all_model_data))**0.5
            mae_value_store.append(mae_values)
            rmse_values_store.append(rmse_values)
            print(mae_values)
            print(rmse_values)

    # store run, model and errors in df

        model_names = all_model_data.columns[2:numcols]
        run_results = pd.DataFrame({
            'Run': [runcount]*len(model_names),
            'Model_name': model_names,
            'MAE': mae_value_store,
            'RMSE': rmse_values_store,
            'Data': target
        })

    # add each test to a df
        model_test_metrics = pd.concat([model_test_metrics, run_results], ignore_index=True)
        end_time = time.perf_counter()
        run_time = end_time - start_time

        print("")
        print("")
        print(f"{target}, column {colcount} of 11, test {runcount} of 2200.")
        print("")
        print(f"Time for test {runcount}: {run_time}")
        print("")
        print("")

# group models by name and calc average MAE and RMSE for all forecasts
model_metrics_final = model_test_metrics.groupby(['Data','Model_name'])[['MAE', 'RMSE']].mean().reset_index()

file_name = f"py_Test_Ave_h{h}_non_opt.csv"
save_dr =f"C:/Users/cb241/OneDrive/Documents/1 - University/MAppFin/Semester H/FINAN521 - Finance Project/Results/{file_name}"
model_metrics_final.to_csv(save_dr)