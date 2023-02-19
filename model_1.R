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
  mutate(month = as.factor(lubridate::month(period))) 
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





# Data exploitation on whole data set===========================================
# Correlation and multicolineariy testing
# Assuming your data frame is named 'df' and the response variable is named 'Y'
par(mfrow=c(3,3))  # Set the plot layout to a 3x3 grid
for (i in 2:10) {  # Assuming you have 8 predictor variables in your data frame
  plot(abs_impute_2[,i], abs_impute_2$Y, main = paste0("Scatterplot of Y vs ", colnames(abs_impute_2)[i]))
}

# Outliers
abs_impute_2_numeric <- abs_impute_2 %>% 
  select(-period, -month) # Remove the non-numeric varibles

# Histograme of all inputs
hist.data.frame(abs_impute_2_numeric)

# Boxplots of all inputs
# Melt the data into long format
library(reshape2)
abs_impute_2_melt <- melt(abs_impute_2_numeric)

# Create a grid of boxplots, one for each variable
ggplot(abs_impute_2_melt, aes(x = variable, y = value)) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free_y") +
  xlab("") +
  theme(axis.text.x = element_blank())




# calculate z-scores for each column
z_scores <- apply(abs_impute_2_numeric, 2, function(x) abs((x - mean(x)) / sd(x)))
z_scores_df <- as.data.frame(z_scores) # Convert to df


# get row indices with any z-score greater than 3
outlier_rows <- which(apply(z_scores, 1, function(x) any(x > 3)))

# create a data frame with outlier rows and corresponding z-scores
outliers <- data.frame(row = outlier_rows, z_scores = apply(z_scores[outlier_rows,], 1, max))
outliers$col <- apply(z_scores[outlier_rows,], 1, function(x) which(x == max(x))) # Col number



# Split data into test/train====================================================
# At this stage we split into test/train and did not scale as this could depend
# on which model we select.
# Further manipulations can be applied to test/train at a later stage depending
# on specific model 
#
# We decdide to train on the full data set at this stage

train_abs <- abs_impute_2 %>% 
  filter(period < "2018-03-01") %>% 
  select(-period)
  
dim(train_abs) # 147, 9
test_abs <- abs_impute_2 %>% 
  filter(period >= "2018-03-01") %>% 
  select(-period)

dim(test_abs) # 11, 9 







# Machine learning model========================================================
# As dealing with a regresion problem wich varies in space and time
# we elected to utilize a regression tree model, which we discussed in wk3.
# Possible models include:
# - Basic regression tree
# - PRIM-Bump hunting
# - Multivariate Adaptive Regresion Spline: This did quite well
# - Bagging
# - Random Forest: This did poorly
# - Boosted trees: EXPLORE

set.seed(1234)
library(caret)

# Define the training control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Train the random forest model using 10-fold cross-validation
model <- train(Y ~ .,
               data = train_abs, method = "rf", trControl = ctrl)

# Print the model
print(model)

# Make predictions on the test set using the cross-validated model
yhat <- predict(model, newdata = test_abs)

# Plot the residuals
res_test <- test_abs$Y - yhat
plot(res_test)

random_forest <- randomForest(Y ~ ., data = train_abs, mtry = 8, importance = T)
res <- train_abs$Y - random_forest$predicted
plot(train_abs$Y, res)
par(mfrow=c(2,1), cex=0.7)
barplot(sort(random_forest$importance[,1], decreasing = TRUE))
barplot(sort(random_forest$importance[,2], decreasing = TRUE))

yhat <- predict(random_forest, newdata = test_abs)
res_test <- test_abs$Y - yhat
plot(res_test)


# Create a MARS model using all the data
mars_model <- earth(Y ~ ., data = train_abs)

yhat <- predict(mars_model, newdata = train_abs)
res <- train_abs$Y - yhat
plot(res)


# Define the boosting model
boost_model <- train(Y ~ ., 
                     data = train_abs, 
                     method = "gbm",
                     trControl = trainControl(method = "cv", 
                                              number = 5, 
                                              verboseIter = TRUE), 
                     tuneGrid = expand.grid(n.trees = c(100, 200, 300),
                                            interaction.depth = c(1, 3, 5),
                                            shrinkage = c(0.01, 0.1, 0.5),
                                            n.minobsinnode = 10))

# Make predictions on the test data
predictions <- predict(boost_model, newdata = test_abs)
res <- test_abs$Y - predictions
plot(res)

# ANN Model=====================================================================

# We will need to hot encode month and normalise everything else which is not
# hot encoded for ANN - month is as factor with 4 levels so use caret

dummy_test <- dummyVars( " ~ .", data = test_abs)
test_abs_ann <- data.frame(predict(dummy_test, newdata = test_abs))

train_abs_ann <- dummyVars(" ~ .", data = train_abs)
train_abs_ann <- data.frame(predict(train_abs_ann, newdata = train_abs))

# We will now scale, except the month cols
cols_to_scale_train <- train_abs_ann[, !(names(train_abs_ann) %in% c("Y","month.3", "month.6", "month.9", "month.12"))]
cols_to_scale_test <- test_abs_ann[, !(names(test_abs_ann) %in% c("Y","month.3", "month.6", "month.9", "month.12"))]

# Use scaled test/train set going forward
scaled_train <- scale(cols_to_scale_train)
train_abs_ann_scaled <- cbind(scaled_train, train_abs_ann[, c("Y","month.3", "month.6", "month.9", "month.12")])

scaled_test <- scale(cols_to_scale_test)
test_abs_ann_scaled <- cbind(scaled_test, test_abs_ann[, c("Y","month.3", "month.6", "month.9", "month.12")])

# Apply ANN Model
n <- names(train_abs_ann_scaled)
func <- as.formula(paste("Y ~", paste(n[n != "Y"],collapse = " + ")))
nn1 <- neuralnet(func,data = train_abs_ann_scaled, hidden = c(15,10))
plot(nn1, rep = "best")

(yhat <- predict(nn1, test_abs_ann_scaled) )
res <- yhat - test_abs_ann_scaled$Y 
plot(res)

# Compare ML against ANN========================================================

