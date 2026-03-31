#Modelling - Holt + ARIMA
#NOTE: Train/test split currently set to 2018 (Can be adjusted later for consistency with evaluation)

library(forecast)

#reading in dataset and cleaning it to create time series object
unemp <- read.csv("unemployment_rate.csv")
unique(unemp$Labour.force.characteristics) #for my info
#extracting unemployment rate info
unemp_rate_row <- unemp[unemp$Labour.force.characteristics == "Unemployment rate", ]
#dropping the label column and converting row to vector
values <- unemp_rate_row[ , -1]
values_vec <- as.numeric(gsub(",", "", as.character(values)))

#creating time series
ts_data <- ts(values_vec, start=c(1997,1), frequency=12)
plot(ts_data)

#we observe a smooth macro trend with a spike during COVID in the year 2020.


#train/test splits 
train <- window(ts_data, end=c(2018,12))
test  <- window(ts_data, start=c(2019,1))
length(train)
length(test)

plot(train, col="blue", main="Train vs Test")
lines(test, col="red")
legend("topright", legend=c("Train","Test"), col=c("blue","red"), lty=1)

#Holt Model
holt_fc <- holt(train, h = length(test))
plot(holt_fc)
lines(test, col = "red")
accuracy(holt_fc, test)


# From above, we observe that the training error is low, indicating a good fit 
# , however, the test RMSE is significantly higher (2.48), thus showing poor 
# forecasting performance. 
# This is mainly due to the model’s inability to capture the sudden spike in 
# unemployment during COVID-19 in 2020. Additionally, the high residual 
# autocorrelation (ACF1 = 0.905) suggests that the model does not fully capture 
# the underlying time dependence in the series.


#ARIMA model
#training set
fit_auto <- auto.arima(train)
summary(fit_auto)

#test set
arima_fc <- forecast(fit_auto, h=length(test))
plot(arima_fc)
lines(test, col="red")
accuracy(arima_fc, test)


# B ased on the RMSE, ARIMA performs slightly better than Holt’s model since it has
# lower RMSE. However, it still struggles to capture the sudden spike during COVID, 
# leading to large prediction errors. 
# 
# The forecasts are relatively flat and fail to adapt to abrupt changes in the data.
# Thus, univariate models may be insufficient motivating the use of ARIMAX models
# with external economic indicators.


#trying a few more models - we need differencing due to non staionarity
fit1 <- Arima(train, order=c(1,1,1))
fit2 <- Arima(train, order=c(2,1,1))
fit3 <- Arima(train, order=c(1,1,2))

#adding extra models as suggested in EDA
fit4 <- Arima(train, order=c(0,1,1))
fit5 <- Arima(train, order=c(1,1,0))

AIC(fit_auto, fit1, fit2, fit3, fit4, fit5)

# We observe that the model selected by auto.arima outperforms all the manual ARIMA
# models fitted above, as it has the lowest AIC (-262.34) 


#trying for another model using 
# >auto.arima(train, stepwise=FALSE, approximation=FALSE)
#gives us ARIMA(1,1,2)(0,0,1)[12] with AIC = 264.88
fit_new <- Arima(train, order=c(1,1,2), seasonal=c(0,0,1))
arima_fc_new <- forecast(fit_new, h=length(test))
plot(arima_fc_new)
lines(test, col="red")
accuracy(arima_fc_new, test)

# 
# By comparing the ARIMA (1,1,1)(1,0,2)[12] and ARIMA(1,1,2)(0,0,1)[12], we 
# observe that first has a higher AIC(-262.34) but lower RMSE(2.14),whereas the 
# latter has a lower AIC (-264.88) and higher RMSE (2.15)
# 
# Thus, the new ARIMA model fits the training data better as indicated by the low AIC, 
# but does not improve prediction accuracy, indicated by higher RMSE. 
# Thus, this suggests overfitting and hence we consider ARIMA (1,1,1)(1,0,2)[12] as
# our final ARIMA model. 
# 
# The original series exhibits non-stationarity due to trend, which is also 
# addressed through first-order differencing (d = 1) in the ARIMA models.



checkresiduals(holt_fc)
checkresiduals(fit_auto)
Box.test(residuals(fit_auto), lag=20, type="Ljung-Box")

# The residual diagnostics for Holt’s method show significant autocorrelation, 
# indicating that the residuals are not random, and thus, says that the model fails
# to capture temporal patterns in the data.
# (Since the p-value is also 0.006 < 0.05, this also implies that we reject the 
# hypothesis that the residuals are random)
# 
# The ARIMA model produces residuals closer to white noise, with most 
# autocorrelations lying within the confidence bounds, thus, indicating that the 
# model captures underlying temporal dependence better than Holt’s method.
# Here the p-value = 0.435 > 0.05 => we do not reject that the residuals are random
# 
# 
# Similarly in the box test, we observe a p-value = 0.5722 > 0.05, thus we arrive 
# at the same conclusion of not rejecting for the ARIMA model
