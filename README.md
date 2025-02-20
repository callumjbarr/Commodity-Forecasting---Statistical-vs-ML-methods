# Commodity-Forecasting---Statistical-vs-ML-methods
Compares forecast accuracy of statistical and ML methods on RBA and ANZ commodity price indices.


Findings:

This study compared the performance of traditional forecasting models and machine learning (ML) models on ANZ and RBA commodity price indices over one-month and four-month forecast horizons. 
The findings indicate that ARIMA variants consistently outperformed naïve benchmarks and delivered the best overall accuracy. In contrast, ML models struggled to consistently outperform the benchmarks, revealing no clear advantage. However, Prophet stood out as the worst-performing ML model, surpassing only the simple average, while NeuralProphet’s improved accuracy over Prophet was confirmed. A key observation was the substantial computational time required by ML models, which did not translate into better forecasting accuracy.
These results suggest that analysts forecasting ANZ and RBA commodity prices should prioritize ARIMA models for their robustness and accuracy. For future research, in-depth optimization of ML models is recommended, particularly using Bayesian methods to automate hyperparameter tuning across models and indexes (AWS, 2025). These automated methods test various parameter combinations to identify the optimal settings which would give them a fairer representation in the study. Exploring longer forecast horizons could also yield valuable insights, as accurate long-term predictions are highly beneficial. Additionally, incorporating exogenous variables, such as macroeconomic indicators or weather data, may enhance predictive performance.
