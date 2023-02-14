#===============================================================================
# The purpose of this script is to explore unemplyoment is Australia over the
# last 20 years. The goal of the script is to compare 2 models ability to 
# detect unemployment in Australia through the use of histoic data 
#===============================================================================

# Load in packages =============================================================
library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(gridExtra)


# Load in data set and define column types in data import=======================
abs <- read_excel("AUS_Data - Copy.xlsx", 
                  col_types = c("date", "numeric", "text", 
                                "numeric", "numeric", "numeric", 
                                "numeric", "numeric", "numeric"))
# Data column name cleaning
abs <- abs[-1,] 
abs$period <- abs$...1
abs$...1 <- NULL

# We will also metric for our analysis to the period 1998. We will split date
# out into month and year as seasonality could be captured by doing so which
# might not be captured otherwise. 
abs_clean_full <- as.data.frame(abs) %>% 
  mutate(year = lubridate::year(period),
         month = lubridate::month(period)) %>% 
  mutate(X1 = as.numeric(X1))

# We will also only retain data from after 1999, as for if we should keep it
# or remove it for ML model it can depend on the context of the research question.
# Does the data before 1999 contain information that could help explain patters?
# Additionally, more data points mean a bigger training sample for the model
abs_clean_1999 <- abs_clean_full %>% 
  filter(year >= 1999)


# Data exploration==============================================================
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
plot(ts(abs_clean_1999[,-c(9,10,11)], start=c(1999,1), frequency=4))
plot(ts(abs_clean_full[,-c(9,10,11)], start=c(1980,1), frequency=4))

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




