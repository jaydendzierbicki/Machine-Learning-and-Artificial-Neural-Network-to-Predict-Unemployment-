# Machine Learning and Artificial Neural Network to Predict Unemployment

## Abstract/Executive Summary
The Australian economy has undergone various macroeconomic events in the past 21 years, impacting the unemployment rate, which is a crucial metric used to guide policy decisions and monitored by the ABS. This study examines the potential of machine learning and neural networks to predict and model unemployment by incorporating macroeconomic indicators, such GDP, terms of trade, CPI, job vacancies, ERP, general government consumption expenditure, and all sectors consumption expenditure. To assess performance, the root mean squared error was used as a metric, and residual plots were examined to compare the performance of these models against a basic naive model. The multivariate adaptive spline regression model (MARS) demonstrated superior performance compared to random forest and boosting models. However, the test RSME was slightly higher than the cross-validation RSME, likely due to modelling the COVID-19 pandemic's impact on unemployment. In addition, a basic neural network with two hidden layers outperformed the MARS model with a lower test RSME; though interestingly had a worse cross validation RMSE. Through increasing the complexity of our neural network to include more neurons we were able to reduce the cross validation RMSE, though MARS still outperformed in this respect. The predictive capability of these models can be utilized to predict unemployment during the waiting period for ABS unemployment reports to be published. However, it is essential to note that unpredictable events such as natural disasters, as observed during COVID-19, could impact the models' predictive power, as they are only as reliable as the training data.

## Introduction - Australia Unemployment: 1999 and 2020
The Australian Bureau of Statistics (ABS) is Australia’s national statistical agency which is responsible for collecting employment statistic such as the unemployment rate, one of many employment statistics. The unemployment rate is a key indicator of the labour market performance, providing a snapshot of the available labour supply at a particular time. The survey is conducted on about 26,000 dwellings, with responses from around 52,000 people every month, ensuring it forms a representative sample of the Australia population, with respondents being asked self-guided questions (How the ABS Measures Unemployment, 2022). There are various limitations of the unemployment rate, with the most obvious being underemployment, with the true unemployment rate tending to be larger than the actual unemployment rate. 
The Australian government often relies on the unemployment rate to shape policy decisions, especially during election cycles (Cockburn et al., 2022; Visentin et al., 2022). High unemployment rates have even been associated with election outcomes (Leigh & McLeish, 2009), which can lead to a change in government. Over the past 21 years, the Australian labour market has experienced various macroeconomic events that have influenced the unemployment rate, which has averaged around 5.6% during this period. After the recession of the 1990s, the labour market started to recover in the early 2000s, with the unemployment rate dropping to around 4.1%. However, the global financial crisis (GFC) of 2007-2008 had a significant impact on the Australian economy, causing the unemployment rate to rise to 5.75% (RBA, 2010). During such economic events, the unemployment rate is often linked to various macroeconomic indicators, such as GDP, with declining GDP leading to an increase in unemployment, as seen during recessions or the GFC (Higgens, 2011). The Covid-19 pandemic in 2019-2020 caused a sharp increase in unemployment to 7.1%, but unlike traditional recessions, the movement of other indicators such as GDP and house prices did not follow a typical recession pattern (Owen, 2022). This paper aims to predict unemployment using various macroeconomic indicators such as GDP, terms of trade, CPI, job vacancies, ERP, general government consumption expenditure, and all sectors consumption expenditure, by training two models on data from 1981 to 2018. A grid search approachwas used to tune the hyperparameters of the models. However, we acknowledge that the models may perform poorly during events such as the Covid-19 pandemic, which deviates from normal economic conditions. Therefore, the ABS survey will still be valuable and necessary for measuring unemployment during such events.
