# Time Series Forecasting with Neural Networks

## Introduction to Time Series Forecasting
Time series: A series of data point in time order
* Spread over a continuous time interval
* Within the continuous interval, time is spread out sequentially (e.g. daily, monthly, quarterly, annually)
* Each time unit within the time interval has, at most, one data point (e.g., monthly sales for a product)

Time series may include general trends like upward, downward, horizontal or no general trend

Seasonality or seasonal pattern indicates regular patterns that occur within a time interval in addition to a general trend (e.g. spike due to winter holidays, dip due to summer)

Cyclical trends indicate patterns across longer periods of time
* A cyclical trend is often confused with seasonality, but there are important distinctions
* Cyclical trends occur across longer periods of time
* They are often spread out over a number of years, opposed to seasonal trends that can occur within a calendar year
* Economic expansions and contractions over decades are good examples of cyclical trends
* Unlike seasonality, cyclical trends do not occur over a fixed or regular time period
* For cyclical trends the length of time for uptrends and downtrends will vary and fluctuate

Irregularities are variability in the data that cannot be explained by the model
* Some time series will exhibit noise or irregularities: variability in the data that cannot be explained by your model
* It may look like jumps or downturns in your data that are not part of a regular or seasonal pattern (e.g., website outage impacting sales)

Time series forecasting is predicting future values of time-dependent data
* e.g., weekly sales, daily inventory levels, hourly website traffic
* Time series forecasting is using a model to analyze the trends and patterns in this historical data to predict the future

Use cases include product demand, workforce demand, financial metrics, inventory sales

Comparison to traditional ML
* Unlike traditional supervised machine learning, the _order_ of the inputted data points matter
* More frequent model retraining
  * greater variability in ground truth of time series forecasting than say identifying a cat with computer vision
    * visual characteristics of cats don't change much overtime but data in a time series is much more likely to change over time (e.g., fashion trends)
* Predictions will be inaccurate so probabilistic forecasting is needed
  * time series forecasting is very difficult to get right and predictions will almost be wrong
  * rely heavily on probabilistic forecasting to generate confidence intervals to provide a range in which future values should fall as opposed to point forecasting
* Little to no historical data
  * use historical data for a similar product but requires handling multiple time series input

## Time Series Forecasting Problem Types
Categories of time series problems:
### 1. One-step-ahead prediction and Multi-step-ahead prediction
Determines how far in the future you'd like to predict

In one-step-ahead prediction, you’re using historical data to make a prediction for the next data point that will occur
* for example, using historical sales data for a particular product to predict tomorrow’s sales of the same product

Multi-step-ahead prediction means that prediction required is farther into the future than just the immediately arriving data point
* for instance, instead of predicting only tomorrow’s sales, you need to predict the product’s sales for the entire month.
### 2. Autoregressive forecasting and Forecasting with covariates
Focused on the types of inputs you are taking in

A time series model that relies solely on observations from its own current and previous time series data is known as autoregressive forecasting
* e.g. predicting an airplane’s future position based solely on its current and previous positions

Forecasting with covariates is when a time series depends on incoming externalities in addition to its recent past states
*  e.g. predicting an airplane's future postiion based on current and previous positions as well as weather and atmospheric variables
### 3. Point forecasting and Probabilistic Forecasting
Differentiate how you want your output presented

Point forecasting returns a single estimate for a time step
* e.g. a predicted stock price of $100 tomorrow

Probabilistic forecasting returns a range of estimates with probabilities for given time step
* Instead of just the one figure prediction (that is, $100), with probabilistic forecasting, you may get a prediction range of $97-$103 with 90% confidence, with $100 being the most probabilistic price at a 50% likelihood
* Logically, the uncertainty around your predictions increases as they move farther into the future
### 4. Univariate time series and Multi-variate / multiple time series
Defines the number of time series the model is working with

Univariate time series makes predictions of data from only one time series (e.g. number of tourists visiting U.S. next year)

Multivariate time series, as the name implies, makes predictions of data from more than one time series
* i.e., instead of solely predicting the number of tourists to visit the U.S. next year, a multivariate time series may also predict the volume of luggage brought with them and their length of stay
* the different time series are treated as a single observation

Multiple time series is similar to the notion of a multivariate time series in that it is making a prediction based on more than one time series; however, it is different from a multivariate time series in that a multiple time series problem treats the different time series as distinct observations
* i.e., a multivariate time series is treating the time series of tourists, their luggage, their lengths of stay, etc. as a single observation in vector form; by contrast, an example of a multiple time series problem would be looking at product sales across an entire company, like Amazon.com
  * each of the millions of products on Amazon.com would have its own distinct time series, which is used to make the forecast

## NNs Solve Challenges with Classical Forecasting Methods
Classical forecasting models fit a single model to each individual time series and then use that model to extrapolate the time series into the future

Examples of classical forecasting models:
* Autoregressive integrated moving average (ARIMA)
* Exponential Smoothing (ETS)

Statistical models struggle with multi time series and no historical data
* multi-time series: time series groupings for demand for different products, server loads, and requests for webpages can benefit from training a single model jointly over all the time series

NN can forecast with multiple related time series
* datasets containing hundreds of related time series are better sutied to NN than ETS and ARIMA
* NN can generate forecasts for new series that are similar to the ones it has been trained on

NN can deal with lots of data and many covariates
* neural networks are good at leveraging long history to learn its influence on future points, and they can handle high dimensionality in the inputs (that is, they can handle many covariates)

NN can learn that order matters, can pursue nonlinearities, and understand complex dependencies
* neural networks can also understand that the order of inputs can matter, learns when that order does matter, and determines the influence of that order
* learn which nonlinearities to pursue
  * makes neural networks less computationally expensive than combinatorial non-linear modeling, which in high-dimensional problems is not a feasible option

## Neural Network Concepts

### Long-Term Dependency Problem
### RNNs
### Vanishing Gradient and LSTM
