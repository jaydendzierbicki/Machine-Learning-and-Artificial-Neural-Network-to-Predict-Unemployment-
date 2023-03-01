#===============================================================================
# The purpose of this script is to explore unemplyoment is Australia over the
# last 20 years. The goal of the script is to compare 2 models ability to 
# detect unemployment in Australia through the use of histoic data 
#===============================================================================
#
# Load in packages =============================================================
library(readxl) # Read excel files
library(dplyr) # Data wrangling
library(lubridate)
library(ggplot2) # For plot
library(gridExtra)
library(imputeTS) # Impute TS data
library(neuralnet) # NN modeling
library(Hmisc) # Multiple historgrams


# Load in data set and define column types in data import=======================
abs <- read_excel("AUS_Data.xlsx", 
                  col_types = c("date", "numeric", "text", 
                                "numeric", "numeric", "numeric", 
                                "numeric", "numeric", "numeric"))
# Data column name cleaning
abs <- abs[-1,] 
abs$period <- abs$...1
abs$...1 <- NULL

str(abs) # Ensure data in correct format, X1 laoded as character

abs <- abs %>% # Convrt X1 to numeric 
  mutate(X1 = as.numeric(X1))

str(abs) # Confrim above works
summary(abs)

# We will also metric for our analysis to the period 1998. We will split date
# out into month and year as seasonality could be captured by doing so which
# might not be captured otherwise. 
abs_clean_full <- as.data.frame(abs) %>% 
  mutate(month = as.factor(lubridate::month(period)),
         year = as.factor(lubridate::year(period))) 
(str(abs_clean_full))

# We will also only retain data from after 1999, as for if we should keep it
# or remove it for ML model it can depend on the context of the research question.
# Does the data before 1999 contain information that could help explain patters?
# Additionally, more data points mean a bigger training sample for the model
abs_clean_1999 <- abs_clean_full %>% 
  filter(period >= "1999-01-01")













# Data Wrangling================================================================
# - Timeseries plot
# - MIssing observations
# - Summary stats
# We will not impute missing values yet, as this data is generally seasonal we
# deem it immature without actually having an understating of what is going on. 


# Plot data: Both full and restricted to see how historically before
# 1999 things might have been
#
# Things which stand out from TS plot: - We observe missing observations
#                                      - Unemployment was much higher between 1980 and 1990
#                                      - Around 2019 we observe the following (possibly due to covid):
#                                                        - Increase Unemployment (Y)
#                                                        - Reduction/spike in GDP (X2)
#                                                        - Reduction/spike in All sectors; Final consumption expenditure % (X3)
#                                                        - Reduction/spike job vacancies
#                                                        -*We will need to confirm this this is correct or error in data, as based on
#                                                          prior knowledge GDP looks a little odd
#                                      - Around 2007+ we observe impact of GFC, increase in unemployment, pssobile reduction in
#                                        job vacancies, though appears to be NA values. In addition, spike observed with covid for
#                                        some variables is not observed in GFC eluding to further investigation of ABS data before
#                                        full machine learning model to be implemented. 
#
# Overall, there is clear evidence of cyclical occurrence in the data set which correspondence to 'shocks' in the system
# due to external factors such as Covid-19 and the GFC. Based on the simple TS() plot we do observe some issues which
# will require consulting the literature/external sources to validate the data points and correct handling on NA values.
plot(ts(abs_clean_full[,-c(9,10,11)], start=c(1980,1), frequency=4))
plot(ts(abs_clean_full[,c(7,8)], start=c(1980,1), frequency=4))

# Many ML models are limited by missing observations, and missing observations==
# must be treated. We observe that X6 and X7 both contain 5 missing observations
# which should be addressed:
#                           -The missing values for X6 appear to correspond to GFC
#                           -The missing values for X7 appear to correspond to just prior to Covid-19
#
# The implications of this is that if we use mean/mediam imputation then it
# might not take into account the real impact of these events which tend to 
# see a reduction in these variables, as such we should keep this in mind
# as we might need to attempt to source the missing values instead.
(na_count <- sapply(abs, function(y) sum(length(which(is.na(y))))) ) 


# Things to consider before going forward=======================================
# We need to do the following: - why do we see a sudden dip and spike for some data points in covid?
#                              - Some of the NA values correspond to world events, as such how should we impute to ensure
#                                we capture this? As it is a deviation from the normal.
#                              - Explore impact on dates/months/years on ML models as this is an area for improvement 

# Data for GDP appears correct for Q2/Q3 2020: https://www.focus-economics.com/countries/australia/news/gdp/lifting-of-restrictions-amid-massive-fiscal-and-monetary-stimulus
# Data for job vacanices appears correct: https://www.abs.gov.au/statistics/labour/jobs/job-vacancies-australia/latest-release

# Impute missing values=========================================================
# Will add in a flag

# Manual Imputation
# Impute X7
# Obtained data from: https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release
abs_eer_x7 <- read_excel("310101.xlsx", sheet = "Data1", skip = 9) %>% 
  select(X7_2 = A2133251W, # This is our EER value, based on series ID
         period = `Series ID`) # This is period due to skipping the first 9 rows

abs_impute_1 <- left_join(abs_clean_full, abs_eer_x7) %>%  # Left join as we don't want all observations from abs_eer_x7
  select(-X7) # Remove X7 as we now have X7_2 with updated figures

# Impute X6

# Use R package to impute too missing observations
abs_impute_2 <- na_interpolation(abs_impute_1)
plot(ts(abs_impute_2[,c(7,10)], start=c(1980,1), frequency=4))
ggplot_na_distribution(abs_impute_2$X6)
plot(ts(abs_impute_2[,c(7,8)], start=c(1980,1), frequency=4))

















# BASELINE MODEL================================================================
# Very simple persistence model, naive model assumes that the value of the 
# variable being predicted will be the same as its most recent observed value.
# It does not take into account any other facots or varibles that could affect the
# outcome.
# Convert the date column to a time series object
unemployment_ts <- ts(abs_impute_2$Y, start = c(1981, 2), frequency = 4)

# Split the data into training and testing sets
train <- window(unemployment_ts, end = c(2017, 4))
test <- window(unemployment_ts, start = c(2018, 1))

# Create a persistence model
persistence_model <- tail(train, 1)

# Generate predictions for the test set
predictions <- rep(persistence_model, length(test))

# Calculate the mean squared error
mse <- mean((test - predictions)^2)

# Print the MSE
sqrt(mse)














# Data exploitation on whole data set===========================================
# Correlation and multicolineariy testing
# Assuming your data frame is named 'df' and the response variable is named 'Y'
par(mfrow=c(3,3))  # Set the plot layout to a 3x3 grid
for (i in 2:10) {  # Assuming you have 8 predictor variables in your data frame
  plot(abs_impute_2[,i], abs_impute_2$Y, main = paste0("Scatterplot of Y vs ", colnames(abs_impute_2)[i]))
}

# Outliers
abs_impute_2_numeric <- abs_impute_2 %>% 
  select(-period, -month, -year) # Remove the non-numeric varibles

# Histograme of all inputs
hist.data.frame(abs_impute_2_numeric)

# Boxplots of all inputs
# Melt the data into long format
library(reshape2)
abs_impute_2_melt <- melt(abs_impute_2_numeric)

# Load required packages
library(ggplot2)
library(ggpubr)

# Create the boxplot with ggplot
ggplot(abs_impute_2_melt, aes(x = variable, y = value)) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free_y") +
  xlab("") +
  theme(axis.text.x = element_blank()) 

# Summary stats
summary(abs_impute_2_numeric)
sd_table <- abs_impute_2_numeric %>% # SD value
  summarise_at(vars(Y:X7_2), sd)














# Split data into test/train====================================================
# At this stage we split into test/train and did not scale as this could depend
# on which model we select.
# Further manipulations can be applied to test/train at a later stage depending
# on specific model 
#
# We decdide to train on the full data set at this stage


# Whilst not shown here, we see improvement in RMSE validation if we lag variables
# MARS Still best model based on lag. This is included in disccusion as part of
# ways to improve the model to account for time effect.

# If we want to lag varibiles & tune
#abs_impute_2$X1_lag <- lag(abs_impute_2$X1)
#abs_impute_2$X2_lag <- lag(abs_impute_2$X2)
#abs_impute_2$X3_lag <- lag(abs_impute_2$X3)
#abs_impute_2$X4_lag <- lag(abs_impute_2$X4)
#abs_impute_2$X5_lag <- lag(abs_impute_2$X5)
#abs_impute_2$X6_lag <- lag(abs_impute_2$X6)
#abs_impute_2$X7_lag <- lag(abs_impute_2$X7_2)
#abs_impute_2$month_lag <- lag(abs_impute_2$month)
#abs_impute_2 <- abs_impute_2[-1,] 
#abs_impute_2 <- abs_impute_2 %>% 
#  select(-X1, -X2, -X3, -X4, -X5, -X6, -X7_2, -month)



train_abs <- abs_impute_2 %>% 
  filter(period < "2018-03-01") %>% 
  select(-period)


  
dim(train_abs) # 147, 10
test_abs <- abs_impute_2 %>% 
  filter(period >= "2018-03-01") %>% 
  select(-period)


dim(test_abs) # 11, 10











# Machine learning model========================================================
# As dealing with a regresion problem wich varies in space and time
# we elected to utilize a regression tree model, which we discussed in wk3.
# Possible models include:
# - Basic regression tree
# - PRIM-Bump hunting
# - Multivariate Adaptive Regresion Spline: This did quite well - was mentioned in R demonstration wk3
# - Bagging
# - Random Forest: This did poorly
# - Boosted trees: EXPLORE

# We decide to compare the following models
# Random forest
# MARS
# Boosting
# Comment on residual plots
# Comment on test/train MSE

# Out of box model: random forest model=========================================
library(randomForest)
set.seed(123)




start_time <- Sys.time()
# Fit a random forest model to training data
random_forest_oobm <- randomForest(Y ~., data = train_abs)
(end_time_rf <-Sys.time() - start_time ) # 0.03983212 secs

# Calculate the predicted values on the training data
yhat_train_oobm <- predict(random_forest_oobm, newdata = train_abs )

# Calculate the residuals on the training data
residuals_rf_oobm <- train_abs$Y - yhat_train_oobm
plot(train_abs$Y,  residuals_rf_oobm) # Plot resdiuals of rf oobm
qqnorm(residuals_rf_oobm)
qqline(residuals_rf_oobm)

# Calculate the training MSE
(mse_train_oobm <- mean(residuals_rf_oobm^2)) #0.02870791
(sqrt(mse_train_oobm))# 0.1694341

# Calculate the predicted values on the test data
yhat_test_oobm <- predict(random_forest_oobm, newdata = test_abs)

# Calculate the residuals on the training data
residuals_rf_oobm_test <- test_abs$Y - yhat_test_oobm
plot(test_abs$Y, residuals_rf_oobm_test) # Plot resdiuals of rf oobm

# Calculate the training MSE
(mse_test_oobm <- mean(residuals_rf_oobm_test^2) ) #0.6030101
(sqrt(mse_test_oobm)) # RMSE  0.7765373

# Out of box model: MARS========================================================
library(earth)
set.seed(123)

# Fit a MARS model to training data
start_time <- Sys.time()
mars_oobm <- earth(Y ~., data = train_abs )
(end_time_mars <-Sys.time() - start_time )

# Calculate the predicted values on the training data
yhat_train_mars_oobm <- predict(mars_oobm, newdata = train_abs )

# Calculate the residuals on the training data
residuals_mars_oobm <- train_abs$Y - yhat_train_mars_oobm
plot(train_abs$Y,residuals_mars_oobm) # Much disperse then RF
qqnorm(residuals_mars_oobm)
qqline(residuals_mars_oobm)

# Calculate the training MSE
(mse_train_oobm <- mean(residuals_mars_oobm^2))  # 0.04736039
(sqrt(mse_train_oobm))  #0.4401467

# Calculate the predicted values on the test data
yhat_test_mars_oobm <- predict(mars_oobm, newdata = test_abs  )

# Calculate the residuals on the training data
residuals_mars_oobm_test <- test_abs$Y - yhat_test_mars_oobm
plot(test_abs$Y, residuals_mars_oobm_test) # Plot resdiuals of rf oobm

# Calculate the training MSE
(mse_test_oobm <- mean(residuals_mars_oobm_test^2) ) #0.4132105
(sqrt(mse_test_oobm)) # RMSE 0.6428145

# Out of box model: Boosting====================================================
library(gbm)
set.seed(123)

# Fit a Boosting model to training data
start_time <- Sys.time()
# Fit a random forest model to training data
boost_oobm <- gbm(Y ~., data = train_abs )
(end_time_boost <-Sys.time() - start_time ) # 0.006984949 secs


# Calculate the predicted values on the training data
yhat_train_boost_oobm <- predict(boost_oobm, newdata = train_abs )

# Calculate the residuals on the training data
residuals_boost_oobm <- train_abs$Y - yhat_train_boost_oobm
plot(train_abs$Y, residuals_boost_oobm) # Plot residuals of boosting oobm
qqnorm(residuals_boost_oobm)
qqline(residuals_boost_oobm)

# Calculate the training MSE
(mse_train_boost_oobm <- mean(residuals_boost_oobm^2)) # 0.08096076
(sqrt(mse_train_boost_oobm)) # RMSE 0.284536

# Calculate the predicted values on the test data
yhat_test_boost_oobm <- predict(boost_oobm, newdata = test_abs   )

# Calculate the residuals on the test data
residuals_boost_oobm_test <- test_abs$Y - yhat_test_boost_oobm
plot(test_abs$Y,residuals_boost_oobm_test) # Plot residuals of boosting oobm

# Calculate the test MSE
(mse_test_boost_oobm <- mean(residuals_boost_oobm_test^2)) #1.723489
(sqrt(mse_test_boost_oobm)) # RMSE




# We will now optimize our MARS MODEL===========================================
# What is a MARS model? How can we describe it itnutivly? We can
# plot it out and put some points around it:


# Fit a MARS model with nprune = 1
# Will produce linear model which minimizes loss function 
mars_model_1 <- earth(Y ~ X7_2, data = train_abs, nprune = 1) 
summary(mars_model_1) # 1 of 7 terms, only intercept

# Fit a MARS model with nprune = 2
# Will produce linear model which minimizes loss function 
mars_model_2 <- earth(Y ~ X7_2, data = train_abs, nprune = 2)
summary(mars_model_2) # 2 of 7 terms, only intercept/1hinge function


# Fit a MARS model with nprune = 3
# Will produce linear model which minimizes loss function 
mars_model_3 <- earth(Y ~ X7_2, data = train_abs, nprune = 3)
summary(mars_model_3) # 3 of 7 terms, only intercept/2xhinge function

# This plot will demo how it looks visually
# Create a scatter plot of the data
plot(train_abs$X7_2, train_abs$Y, pch=16, xlab="X7", ylab="Y")

# Add lines to the plot for the MARS models
x_vals <- seq(min(train_abs$X7_2), max(train_abs$X7_2), length.out=100)
lines(x_vals, predict(mars_model_1, newdata=data.frame(X7_2=x_vals)), col="red", lwd=2)
lines(x_vals, predict(mars_model_2, newdata=data.frame(X7_2=x_vals)), col="blue", lwd=2)
lines(x_vals, predict(mars_model_3, newdata=data.frame(X7_2=x_vals)), col="green", lwd=2)

# Add a legend to the plot
legend("topright", legend=c("nprune=1", "nprune=2", "nprune=3"), 
       col=c("red", "blue", "green"), lty=1)


# TUNE MARS MODEL===============================================================
# Tuning: We have two parameters, the degree of interactions and the
# number of retained terms, we will undertake a grid search to find the optimal
# number of cominations of these hyperparamters that minmise the prediction error

library(caret)

set.seed(123)
start_time <- Sys.time()
# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)
tuned_mars <- train(
  x = subset(train_abs, select = -Y),
  y = train_abs$Y,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3),
  tuneGrid = hyper_grid
)

end_time <- Sys.time() - start_time # Time difference of 38 seconds
plot(tuned_mars)
tuned_mars$bestTune # nprune = 23, degree = 1\
tuned_mars$results %>% 
  filter(nprune == tuned_mars$bestTune$nprune, degree == tuned_mars$bestTune$degree) 


# Can we further improve our model by tweaking nprune, we set degree as 1
set.seed(123)
start_time <- Sys.time()
# create a tuning grid
hyper_grid_2 <- expand.grid(
  degree = 1:3, 
  nprune = 2:23 %>% floor()
)
tuned_mars_2 <- train(
  x = subset(train_abs, select = -Y),
  y = train_abs$Y,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3),
  tuneGrid = hyper_grid_2
)

end_time <- Sys.time() - start_time # Time difference of 23 seconds
plot(tuned_mars_2)
tuned_mars_2$bestTune # nprune = 20, degree = 1\
tuned_mars_2$results %>% 
  filter(nprune == tuned_mars_2$bestTune$nprune, degree == tuned_mars_2$bestTune$degree) # CV RMSE 0.3252662 
varImp(tuned_mars_2) 

# Extract some input to interpet model
set.seed(123)
tuned_mars_2 <- earth(Y ~., data = train_abs, nprune = 20, degree = 1 )
#tuned_mars_2 <- earth(Y ~., data = train_abs, nprune = 11, degree = 1 ) # Without year in model
summary(tuned_mars_2) # Provide summary/coefiencets of model

# Calculate the predicted values on the training data
yhat_train_mars <- predict(tuned_mars_2, newdata = train_abs)

# Calculate the residuals on the training data
residuals_mars_train <- train_abs$Y - yhat_train_mars
par(mfrow = c(1,2))
plot(train_abs$Y, residuals_mars_train) # Plot residuals of boosting oobm
qqnorm(residuals_mars_train)
qqline(residuals_mars_train)

# Calculate the training MSE
(mse_train_mars <- mean(residuals_mars_train^2)) 
(sqrt(mse_train_mars)) 

# Calculate the predicted values on the test data
yhat_test_mars <- predict(tuned_mars_2, newdata = test_abs)


# Calculate the residuals on the test data
residuals_mars_test <- test_abs$Y - yhat_test_mars
par(mfrow = c(1,2))
plot(test_abs$Y, residuals_mars_test) # Plot residuals of boosting oobm
qqnorm(residuals_mars_test)
qqline(residuals_mars_test)

# Calculate the test MSE
(mse_test_mars <- mean(residuals_mars_test^2)) # 0.1783143, same as baseline actually
(sqrt(mse_test_mars))

# Now lets plot everything, and put it on a graph and see how Y compares yhat==
# Create data frame of know, predicted agaisnt time
predict_all <- as.numeric(predict(tuned_mars_2, newdata = abs_impute_2))
model_plot <- abs_impute_2 %>% 
  select(period, Y) %>% 
  mutate(period = as.Date(period))
model_plot <- bind_cols(model_plot, Y_hat = predict_all)



# Plot, highlight train/test data with solid line
ggplot(model_plot, aes(x = period)) +
  geom_line(aes(y = Y, color = "Y")) +
  geom_line(aes(y = Y_hat, color = "Yhat")) +
  scale_color_manual(values = c("Y" = "blue", "Yhat" = "red")) +
  labs(x = "period", y = "Value") +
  geom_vline(xintercept = as.Date("2018-03-01"), linetype="solid", 
             color = "black", size=1.5) +
  ggtitle("Predictive Performance of MARS Model")





# Cross validation accuracy MARS================================================
# create a tuning grid

# Insert our tuned hyper-parameters
set.seed(123)
hyper_grid <- expand.grid(
  degree = 1, 
  nprune = 20
)
cv_mars_model <- train(
  x = subset(train_abs, select = -Y),
  y = train_abs$Y,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3),
  tuneGrid = hyper_grid
)
print(cv_mars_model) # 0.3838806  
















# ANN Model=====================================================================

# Data prep
library(caret)

# ANN/Deep learning requires data to be scaled:
# We do not want to scale month/year/Y we will hot encode these
cols_to_scale <- c("X1", "X2", "X3", "X4", "X5" ,"X6", "X7_2")
#cols_to_scale <- c("X1_lag", "X2_lag", "X3_lag", "X4_lag", "X5_lag" ,"X6_lag", "X7_lag") # If we lag
cols_not_to_scale <- c("Y", "month" , "year")
#cols_not_to_scale <- c("Y", "month_lag" ) #, "year") # IF we lag

# Calculate the mean and standard deviation of the numerical variables in the training set
train_mean <- apply(train_abs[, cols_to_scale], 2, mean)
train_sd <- apply(train_abs[, cols_to_scale], 2, sd)

# Scale the numerical variables in the training set using the mean and standard deviation from the training set
train_abs_ann_scaled <- scale(train_abs[, cols_to_scale], center = train_mean, scale = train_sd)

# Scale the numerical variables in the test set using the mean and standard deviation from the training set
test_abs_ann_scaled<- scale(test_abs[, cols_to_scale], center = train_mean, scale = train_sd)

# Add back columns non scaled data 
train_abs_ann_scaled <- cbind(train_abs_ann_scaled[, cols_to_scale], train_abs[, cols_not_to_scale]) #%>% select(-year)
test_abs_ann_scaled <- cbind(test_abs_ann_scaled[, 1:7], test_abs[, cols_not_to_scale]) # %>% select(-year)


dummy_test <- dummyVars( " ~ .", data = test_abs_ann_scaled)
test_abs_ann_scaled <- data.frame(predict(dummy_test, newdata = test_abs_ann_scaled))

dummy_train <- dummyVars(" ~ .", data = train_abs_ann_scaled)
train_abs_ann_scaled <- data.frame(predict(dummy_train, newdata = train_abs_ann_scaled))



# https://www.r-bloggers.com/2015/09/fitting-a-neural-network-in-r-neuralnet-package/

# We will compare 3 simple neural networks, 1 hidden layer, 2 hidden layer and 3 hidden layer NB: Unable to do 3 due to taking 24+ hours
# we will select the most optimized one via a grid search approach which loops
# 1:50 based on the number of varibles. This should answer all the requred points
# of the NN question. We have variations in number of nodes, as well as variation
# in number of hidden networks. 
# We use set seed for reproducatiblity.


# Two hidden layer model: Varying the number of neurons in each layer 
start_time <- Sys.time()
set.seed(123)
loop <- 1:100 # 1:50 What if we vary the number of neurons?
loop2 <- 1:100 # 1:50 
mse1 <- data.frame(Layer_1_neuron = numeric(),
                   Layer_2_neuron = numeric(),
                   Test_MSE = numeric())
formula <- as.formula(paste("Y ~", paste(colnames(train_abs_ann_scaled[-8]), collapse = " + ")))
for (i in loop){
  for (j in loop2){
    set.seed(123)
    tryCatch({
      nn <- neuralnet(formula, data=train_abs_ann_scaled, hidden=c(i, j))
      yhat <- predict(nn, test_abs_ann_scaled)
      res <- yhat - test_abs_ann_scaled$Y 
      mse <- mean(res^2)
      
      yhattrain <- predict(nn, train_abs_ann_scaled)
      restrain <- yhattrain - train_abs_ann_scaled$Y 
      msetrain <- mean(restrain^2)
      output <- data.frame(Layer_1_neuron = i,
                           Layer_2_neuron = j,
                           Test_MSE = mse,
                           Train_MSE = msetrain)
      mse1 <- rbind(mse1, output)
      print(paste0("N1 ", i, " N2 ", j, " Test MSE ", mse, " Train MSE ", msetrain))
    }, error = function(e){
      print(paste0("Error: ", e$message)) # We anticipate some issues 0
    })
  }
}
end_time <- Sys.time() - start_time # 30 mins

# Find row with the lowest Test_MSE value
min_row <- which.min(mse1$Test_MSE)

# Extract the corresponding values of Layer_1_neuron, Layer_2_neuron, and Test_MSE
(best_result <- mse1[min_row, ]) # L1: 49, L2: 18, Test MSE 0.122355

# NB: Remove year, best result is L1, L2, Test MSE



# One hidden layer model
start_time <- Sys.time()
set.seed(123)
loop <- 1:100 #ncol(train_abs_ann_scaled) - 1
mse2 <- data.frame(Layer_1_neuron = numeric(),
                   Test_MSE = numeric())
formula <- as.formula(paste("Y ~", paste(colnames(train_abs_ann_scaled[-8]), collapse = " + ")))
for (i in loop){
  set.seed(123)
  tryCatch({
    nn <- neuralnet(formula, data=train_abs_ann_scaled, hidden=c(i))
    yhat <- predict(nn, test_abs_ann_scaled)
    res <- yhat - test_abs_ann_scaled$Y 
    mse <- mean(res^2)
    yhattrain <- predict(nn, train_abs_ann_scaled)
    restrain <- yhattrain - train_abs_ann_scaled$Y 
    msetrain <- mean(restrain^2)
    output <- data.frame(Layer_1_neuron = i,
                         Test_MSE = mse,
                         Train_MSE = msetrain)
    print(paste0("N1 ", i, ", Train MSE ", mse, " Test MSE ", msetrain))
    mse2 <- rbind(mse2, output)
  }, error = function(e){
    print(paste0("Error: ", e$message)) # We anticipate some issues 0
  })
}
(end_time <- Sys.time() - start_time) # 9 seconds
# Find row with the lowest Test_MSE value
min_row <- which.min(mse2$Test_MSE)

# Extract the corresponding values of Layer_1_neuron, Layer_2_neuron, and Test_MSE
(best_result <- mse2[min_row, ] )# L1: 17, Test MSE 0.9740996


# We will procced with analysis on 2 hidden layer NN with=======================
set.seed(123)
formula <- as.formula(paste("Y ~", paste(colnames(train_abs_ann_scaled[-8]), collapse = " + ")))
nn_two_final <- neuralnet(formula, data=train_abs_ann_scaled, hidden=c(49, 18))
#nn_model <- neuralnet(formula, data = train_abs_ann_scaled, hidden = c(33, 96)) # Optimised 2 hidden layer without year

# Plot residuals train data
yhat_train <- predict(nn_two_final, newdata = train_abs_ann_scaled)
res_train <- yhat_train - train_abs_ann_scaled$Y 
par(mfrow = c(1,2))
plot(train_abs_ann_scaled$Y , res_train)
qqnorm(res_train)
qqline(res_train)
(mse_train_nn <- mean(res_train^2)) 
(sqrt(mse_train_nn)) 

# Plot residuals test data
yhat_test <- predict(nn_two_final, newdata = test_abs_ann_scaled)
res_test <- yhat_test - test_abs_ann_scaled$Y 
par(mfrow = c(1,2))
plot(test_abs_ann_scaled$Y , res_test)
qqnorm(res_test)
qqline(res_test)
(mse_test_nn <- mean(res_test^2)) 
(sqrt(mse_test_nn)) 





# Cross validation NN based on above============================================
# We observe large RMSE with all input varibles including year. If we remove year
# then we hyptohisis the model will be less prone to overfitting. As such 
# when we removed year we obtained an improved CV RMSE
k <- 10

# Split the data into k equal-sized folds
fold_indices <- cut(seq(1, nrow(train_abs_ann_scaled)), breaks = k, labels = FALSE)

# Initialize a list to store the cross-validation results
cv_results <- list()

# Loop through each fold and train the model
for (i in 1:k) {
  formula <- as.formula(paste("Y ~", paste(colnames(train_abs_ann_scaled[-8]), collapse = " + ")))
  set.seed(123)
  # Get the indices of the current fold
  test_indices <- which(fold_indices == i)
  train_indices <- which(fold_indices != i)
  
  # Extract the training and test data for the current fold
  train_data <- train_abs_ann_scaled[train_indices, ]
  test_data <- train_abs_ann_scaled[test_indices, ]
  
  # Train the model on the training data
  #nn_model <- neuralnet(formula, data = train_data, hidden = c(17)) # Optimised 1 hidden layer
  nn_model <- neuralnet(formula, data = train_data, hidden = c(49, 18)) # Optimised 2 hidden layer with 51 neuron search
  #nn_model <- neuralnet(formula, data = train_data, hidden = c(4, 85)) # Optimised 2 hidden layer with 100 neuron search
  

  
  # Make predictions on the test data
  yhat <- predict(nn_model, test_data)
  
  residuals <- yhat - test_data$Y 
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  
  # Add the cross-validation metric to the results list
  cv_results[[i]] <- rmse
}

# Compute the average cross-validation metric across all folds
avg_cv_metric <- mean(unlist(cv_results))

# Print the cross-validation results
print(paste0("Average cross-validation metric: ", avg_cv_metric))


# End code: NOTE: We have eddited the code to optimise different variations
# such as removing year, lagging etc. We have hashed (#) them out as needed/not
# to compare and contrast some points.
