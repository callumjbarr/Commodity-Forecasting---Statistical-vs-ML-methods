#code draws heavily on: https://www.appsilon.com/post/r-time-series-forecasting

################################################################################
# Library install and load
################################################################################
library(tseries)
library(zoo)
library(forecast)
library(neuralprophet)
library(tidyverse)
library(modeltime)
library(modeltime.gluonts)
library(timetk)
library(tidymodels)
library(reticulate)
library(nixtlar)
library(forecast)
library(Metrics)
library(prophet)
library(dplyr)

################################################################################
# Data import
################################################################################

wDir <- "C:\\Users\\cb241\\OneDrive\\Documents\\1 - University\\MAppFin\\Semester H\\FINAN521 - Finance Project\\Data"
setwd(wDir)
dataDf <- read.csv("FINAN521 - Dataset - Final.csv")
dataDf$date <- as.Date(dataDf$date, format = "%d/%m/%Y")
dim(dataDf)
nixtla_set_api_key(api_key = "nixak-WjXqWXzZIAyqRW0S7UUYKf5UBQ8ccND5Z8wy5VTz7oVSAYQ3iqjMfuF4qMh6fMqidnhTri9H9luv8ptz")

################################################################################
# Forecasting models
################################################################################

# periods forecast ahead
h <- 1  
runcount <- 0

# initializing object to store tests
model_test_set_metrics_final <- NULL

# setting min training length and finding max of data to set upper limit
numTest <- 3
lenData <- nrow(dataDf)
minTrainLen <- lenData - numTest 

#loop through all dataset columns
for (icol in 2:dim(dataDf)[2]) {
  
# loop through from min train length adding 1 until length of data minus 
# forecast horizon (so can still calculate MAE and RMSE)
for (j in minTrainLen:(lenData-h) ) {
  
  print(j)
  train_set <- dataDf[1:j, ]
  test_set <- dataDf[(j+1):(j+h), ]
  dim(train_set)
  dim(test_set)
  dataDf$date <- as.Date(dataDf$date)
  
  # set target variable
  target_name <- colnames(dataDf)[icol]
  train_set$target<-train_set[[target_name]]
  test_set$target<-test_set[[target_name]]
    targetDf_train <- train_set %>%
    select(date, target)
  
  # model 1: simple average
    train_set_avg <- mean(train_set$target)
    simple_avg_predictions <- data.frame(
    date = test_set$date,
    target = rep(train_set_avg, nrow(test_set)))
   
  # Naive1
    naive_predictions <- data.frame(
    date = test_set$date, 
    target = rep(last(train_set$target), nrow(test_set)))
  
  # Naive 2 
  train_avg_diff <- mean(diff(train_set$target))
  rwdrift_predictions <- data.frame(
    date = test_set$date, 
    target = last(train_set$target) + train_avg_diff * seq_len(nrow(test_set)))

  # model 2: moving averages
  series_ts <- ts(train_set$target, start = c(1986, 01), frequency = 12)
  ma3 <- ma(series_ts, order = 3, centre = FALSE)
  ma6 <- ma(series_ts, order = 6, centre = FALSE)
  ma12 <- ma(series_ts, order = 12, centre = FALSE)
  ma3_forecast <- forecast(ma3, h = nrow(test_set))
  ma6_forecast <- forecast(ma6, h = nrow(test_set))
  ma12_forecast <- forecast(ma12, h = nrow(test_set))
  ma_forecast_df <- data.frame(
    date = test_set$date,
    MA3 = ma3_forecast$mean,
    MA6 = ma6_forecast$mean,
    MA12 = ma12_forecast$mean)
  
  # model 3: simple exponential smoothing
  ses <- HoltWinters(series_ts, beta = FALSE, gamma = FALSE)
  des <- HoltWinters(series_ts, gamma = FALSE)
  ses_forecast <- forecast(ses, h = nrow(test_set))
  des_forecast <- forecast(des, h = nrow(test_set))
  
  # getting error
  tes_forecast <- tryCatch({
    tes <- HoltWinters(series_ts)
    forecast(tes, h = nrow(test_set))
  }, error = function(e){
    cat("Error in HoltWinters:", e$message, "\n")
    list(mean = rep(NA, nrow(test_set)))
      # Return NA values for the forecast
  })
  
  exsm_forecast_df <- data.frame(
    date = test_set$date,
    SES = ses_forecast$mean,
    DES = des_forecast$mean,
    TES = tes_forecast$mean)
  
  # model 3:  exponential smoothing with seasonal patterns
  tes_seasonal_add_forecast <- tryCatch({
    tes_seasonal_add <- HoltWinters(series_ts, seasonal = "additive")
    forecast(tes_seasonal_add, h = nrow(test_set))
  }, error = function(e){
    cat("Error in HoltWinters:", e$message, "\n")
    list(mean = rep(NA, nrow(test_set)))  # Return NA values for the forecast
  })
  
  tes_seasonal_mul_forecast <- tryCatch({
    tes_seasonal_mul <- HoltWinters(series_ts, seasonal = "multiplicative")
    forecast(tes_seasonal_mul, h = nrow(test_set))
  }, error = function(e){
    cat("Error in HoltWinters:", e$message, "\n")
    list(mean = rep(NA, nrow(test_set)))  # Return NA values for the forecast
  })
  
  exsm_tes_forecast_df <- data.frame(
    date = test_set$date,
    TES = tes_forecast$mean,
    TESAdd = tes_seasonal_add_forecast$mean,
    TESMul = tes_seasonal_mul_forecast$mean)
  
  # AR and ARMA Model
  ar_model <- Arima(series_ts, order = c(1, 0, 0), method = "CSS")
  arma_model <- Arima(series_ts, order = c(1, 0, 1), method = "CSS")
  
  # ARIMA(1,1,1) model
  arima_forecasts <- tryCatch({
    arima_model <- Arima(series_ts, order = c(1, 1, 1))
    forecast(arima_model, h = nrow(test_set))
  }, error = function(e){
    list(mean = rep(NA, nrow(test_set)))
  })
  
  auto_arima_no_season_model <- auto.arima(series_ts, seasonal = FALSE)
  auto_arima_season_model <- auto.arima(series_ts, seasonal = TRUE)
  ar_forecasts <- forecast(ar_model, h = nrow(test_set)) 
  arma_forecasts <- forecast(arma_model, h = nrow(test_set))
  auto_arima_no_season_forecasts <- forecast(auto_arima_no_season_model, h = nrow(test_set))
  auto_arima_season_forecasts <- forecast(auto_arima_season_model, h = nrow(test_set))
  
  # arima models
  arima_forcast_df <- data.frame(
    date = test_set$date,
    AR = ar_forecasts$mean,
    ARMA = arma_forecasts$mean,
    ARIMA = arima_forecasts$mean,
    AutoARIMANoSeason = auto_arima_no_season_forecasts$mean,
    AutoARIMASeason = auto_arima_season_forecasts$mean
  )
  
  # Prophet model
  df <- data.frame(ds = train_set$date, y = as.vector(train_set$target))
  model_prophet <-  prophet(df, 
    changepoint.prior.scale = 0.2,
    changepoint.range = 0.9)
  
  test_setdf <- data.frame(ds = test_set$date, y = as.vector(test_set$target))
  prophet_forecast <- predict(model_prophet, test_setdf)
  
  # TimeGPT Model
  timeGPTdf <- df%>%
    mutate(unique_id = "series_1")
  timeGPT_Forecast <- nixtla_client_forecast(timeGPTdf, 
    h = nrow(test_set),
    freq = "M",
    finetune_loss = 'mae',
    finetune_steps = 50,)
  
  ################################################################################
  # Combining forecasts into single dataframe
  ################################################################################
  
  all_model_data <- data.frame(
    date = test_set$date,
    Actual =  test_set$target,
    AVG = simple_avg_predictions$target,
    Naive1 = naive_predictions$target,
    Naive2 = rwdrift_predictions$target,
    MA3 = ma3_forecast$mean,
    MA6 = ma6_forecast$mean,
    MA12 = ma12_forecast$mean,
    SES = ses_forecast$mean,
    DES = des_forecast$mean,
    TES = tes_forecast$mean,
    TESAdd = tes_seasonal_add_forecast$mean,
    TESMul = tes_seasonal_mul_forecast$mean,
    AR = ar_forecasts$mean,
    ARMA = arma_forecasts$mean,
    ARIMA = arima_forecasts$mean,
    AutoARIMANoSeason = auto_arima_no_season_forecasts$mean,
    AutoARIMASeason = auto_arima_season_forecasts$mean,
    prophet_forecast  = prophet_forecast$yhat,
    TimeGPT_forecast = timeGPT_Forecast$TimeGPT)
  
  # only want the forecast periods data
  all_model_data <- all_model_data[1:h, ]
  dim(all_model_data)
  nmodels <- dim(all_model_data)[2]
  
  ################################################################################
  # calculating forecast accuracy
  ################################################################################
  
  # vector to store error calcs
  mae_values_store <- c()
  rmse_values_store <- c()
  
  # start in column 3 - this is to skip the date and actual price columns
  # for each column calc MAE and RMSE
  for (nmodels_tic in 3:nmodels){
    mae_values <-   sum(abs(all_model_data$Actual-all_model_data[ , nmodels_tic]))/dim(all_model_data)[1]
    rmse_values <-   sum(( ((all_model_data$Actual-all_model_data[ , nmodels_tic])^2))/dim(all_model_data)[1] )^0.5
    
    mae_values_store[nmodels_tic-2 ] <-mae_values 
    rmse_values_store[nmodels_tic-2 ] <-rmse_values 
  }
  
 # add run to dataframe 
  runcount <- runcount +1
  model_test_set_metrics <- data.frame(
    Run = runcount,
    Model_name = names(all_model_data[ ,3:nmodels]),
    MAE = mae_values_store,
    RMSE = rmse_values_store,
    Data = target_name
  )
  
  # indicate how many tests have been completed
  cat("\n")
  print(paste(target_name,"(column ",(icol-1),"of 11), ", "Test", runcount, "of 2200"))
  model_test_set_metrics_final <- rbind(model_test_set_metrics_final,model_test_set_metrics )
  
}
}
warnings()

# print results summary
model_test_set_metrics_final %>% group_by(Model_name, Data) %>%
  summarise(mean(MAE, na.rm = TRUE), mean(RMSE, na.rm = TRUE))

# save final results in df
test_Total_averages <- as.data.frame(model_test_set_metrics_final %>% group_by(Data, Model_name) %>%
                                       summarise(mean_MAE = mean(MAE, na.rm = TRUE), mean_RMSE = mean(RMSE, na.rm = TRUE) ))

# save result as .csv  
horiz = as.character(h)  
file_name <- paste0("R_Test_Ave_h",horiz,"_opt1.csv")
wdr_file_name <- paste0("C:\\Users\\cb241\\OneDrive\\Documents\\1 - University\\MAppFin\\Semester H\\FINAN521 - Finance Project\\Results\\",file_name)
print(wdr_file_name)
write.csv(test_Total_averages, file = wdr_file_name, row.names = FALSE)
  
