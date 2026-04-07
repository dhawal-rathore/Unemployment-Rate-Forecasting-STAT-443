library(forecast)

unemp <- read.csv("eda_data.csv")
tail(unemp)

unemp_ts <- ts(unemp$unemployment_rate, start = c(1997, 1), frequency = 12)

# Use log of gdp as it's an economic indicator
unemp$log_gdp <- log(unemp$all_gdp)

# The high correlation rate between log_gdp and total_employment means we can chose to ignore total_employment to train models faster.
xreg_ts <- ts(unemp[, c("log_gdp")],
              start = c(1997, 1), frequency = 12)

# Test/Train split

unemp_train <- window(unemp_ts, end=c(2018,12))
unemp_test  <- window(unemp_ts, start=c(2019,1))

xreg_train <- window(xreg_ts, end=c(2018,12))
xreg_test  <- window(xreg_ts, start=c(2019,1))

length(unemp_train)
length(unemp_test)

# Plot log GDP

plot(xreg_ts, col = "black", main = "Employment: Train vs Test", ylab = "log_gdp")
lines(xreg_train, col = "blue", lwd = 2)
lines(xreg_test, col = "red", lwd = 2)
legend("topleft", legend = c("Train", "Test"), col = c("blue", "red"), lty = 1, lwd = 2)

# Auto ARMAX model

armax_model_auto  <- auto.arima(unemp_train, xreg = xreg_train,
                           seasonal = TRUE,
                           stepwise = FALSE,
                           approximation = FALSE)

summary(armax_model_auto)

armax_fc_auto <- forecast(armax_model_auto, xreg = xreg_test)

plot(armax_fc_auto,
     main = "ARMAX Forecast: Unemployment Rate",
     ylab = "Unemployment Rate (%)",
     xlab = "Time",
     sub = "Figure 5: ARIMAX forecast vs. actual unemployment rate")

lines(unemp_test, col = "red")
accuracy(armax_fc_auto, unemp_test)

checkresiduals(armax_fc_auto)
Box.test(residuals(armax_model_auto), lag = 20, type = "Ljung-Box")

# Since our p-value for the auto fitted ARMAX model is 0.7325 > 0.05, show our model has captured most mathematical patterns in the data. This is also apparent as our residuals are mostly random noise (except for a few large spikes).
# Like this ARIMA model, the ARMAX model too failed to perform well for the large spike during 2020, but the 95% confidence interval usually captures the true value of unemployment percentage.

# More models with slight variation to see if they perform better on out of sample data.

model_1 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 1), seasonal = c(0, 0, 1), include.drift = TRUE)
model_2 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 0), seasonal = c(0, 0, 1), include.drift = TRUE)
model_3 <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 0), seasonal = c(0, 0, 1), include.drift = TRUE)
model_4 <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 1), seasonal = c(0, 0, 1), include.drift = TRUE)
model_5 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 2), seasonal = c(0, 0, 1), include.drift = TRUE)

model_6  <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 1), seasonal = c(0, 1, 1))
model_7  <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 0), seasonal = c(0, 1, 1))
model_8  <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 0), seasonal = c(0, 1, 1))
model_9  <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 1), seasonal = c(0, 1, 1))
model_10 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 2), seasonal = c(0, 1, 1))
# model_11 is the original baseline model
model_11 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 1), seasonal = c(0, 0, 2), include.drift = TRUE)
model_12 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 0), seasonal = c(0, 0, 2), include.drift = TRUE)
model_13 <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 0), seasonal = c(0, 0, 2), include.drift = TRUE)
model_14 <- Arima(unemp_train, xreg = xreg_train, order = c(1, 1, 1), seasonal = c(0, 0, 2), include.drift = TRUE)
model_15 <- Arima(unemp_train, xreg = xreg_train, order = c(0, 1, 2), seasonal = c(0, 0, 2), include.drift = TRUE)

armax_fc_1  <- forecast(model_1,  xreg = xreg_test)
armax_fc_2  <- forecast(model_2,  xreg = xreg_test)
armax_fc_3  <- forecast(model_3,  xreg = xreg_test)
armax_fc_4  <- forecast(model_4,  xreg = xreg_test)
armax_fc_5  <- forecast(model_5,  xreg = xreg_test)

armax_fc_6  <- forecast(model_6,  xreg = xreg_test)
armax_fc_7  <- forecast(model_7,  xreg = xreg_test)
armax_fc_8  <- forecast(model_8,  xreg = xreg_test)
armax_fc_9  <- forecast(model_9,  xreg = xreg_test)
armax_fc_10 <- forecast(model_10, xreg = xreg_test)

armax_fc_11 <- forecast(model_11, xreg = xreg_test)
armax_fc_12 <- forecast(model_12, xreg = xreg_test)
armax_fc_13 <- forecast(model_13, xreg = xreg_test)
armax_fc_14 <- forecast(model_14, xreg = xreg_test)
armax_fc_15 <- forecast(model_15, xreg = xreg_test)

accuracy(armax_fc_1,  unemp_test)
accuracy(armax_fc_2,  unemp_test)
accuracy(armax_fc_3,  unemp_test)
accuracy(armax_fc_4,  unemp_test)
accuracy(armax_fc_5,  unemp_test)

accuracy(armax_fc_6,  unemp_test)
accuracy(armax_fc_7,  unemp_test)
accuracy(armax_fc_8,  unemp_test)
accuracy(armax_fc_9,  unemp_test)
accuracy(armax_fc_10, unemp_test)

accuracy(armax_fc_11, unemp_test)
accuracy(armax_fc_12, unemp_test)
accuracy(armax_fc_13, unemp_test)
accuracy(armax_fc_14, unemp_test)
accuracy(armax_fc_15, unemp_test)

# Model 15 is the closest

plot(armax_fc_15,
     main = "ARMAX (0,1,2) Forecast: Unemployment Rate",
     ylab = "Unemployment Rate (%)",
     xlab = "Time",
     sub = "Figure 5: ARIMAX forecast vs. actual unemployment rate")

lines(unemp_test, col = "red")

checkresiduals(armax_fc_15)
Box.test(residuals(armax_fc_15), lag = 20, type = "Ljung-Box")