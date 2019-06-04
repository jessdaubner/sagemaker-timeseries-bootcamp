# Module IV - LSTNet

## LSTNet Background
Based on research paper from Carnegie Mellon University by Guokun Lai, Yiming Yang, Wei-Cheng Chang, and Hanxiao Liu - [Modeling Long and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/pdf/1703.07015.pdf)

### Forecasting Challenges
In time series forecasting, your model identifies short-term trends
* Multivariate time series data are found everywhere in our daily lives and relate to a lot of activities such as weather patterns, traffic data, stock market trends, and even tidal movements
* e.g., short-term weekday traffic patterns highlight AM and PM rush hour
    * In order to get an accurate forecasting model, you need a data sample of both the weekday and the weekend patterns over time
    * If the model only looked at this weekday trend, it would predict the same trend for the weekend as well, which would be an inaccurate forecast are quite different for Saturdays and Sundays
* Traditional approaches to forecasting real-world problems like these use an auto-regressive or Gaussian process, which might fail due to this mixture of both weekend and weekday trends.

Considering both long-term and short-term patterns
* The traffic pattern example that exhibits a weekday high volume trend and weekend low volume trend is an example of the long-term dependency problem
* If the model takes a broader view of this data, it will then recognize both the short-term trend and the long-term trend. For the latter, you can see that the trend is to have higher traffic on the weekdays, especially during rush hour periods, and lower, less volatile, traffic on the weekends.
* Accounting for both of these patterns helps to make more accurate forecasts

Forecast based on a multitude of different elements
* It's likely that other variables impact an accurate prediction of traffic volume, like holidays, construction, weather, etc.
  *  Weather, for example, is not simple to predict because there are so many elements to consider when creating the forecast
* This is a challenge of forecasting with autoregressive models

Scale of output is not sensitive to the scale of inputs
* Significant difference in traffic rates between weekdays and weekends
  * Due to significant spread between two main patterns (weekday vs weekend), forecasting over time will likely be inaccurate
* Due to the non-linear nature of the convolutional and recurrent components, one major drawback of the neural network model is that the scale of outputs is not sensitive to the scale of inputs
* Unfortunately, in specific real datasets, the scale of input signals constantly changes in a non-periodic manner, which significantly lowers the forecasting accuracy of the neural network model
* Need some way to scale them so that the data is easier to process for accurate forecasting.

### How LSTNet Solves These Challenges
LSTNet offers a solution to these challenges
* **Long- and Short-term Time Series Network (LSNet)**: A deep learning framework designed to capture a mix of long- and short-term patterns in data for multi-variate time-series
* Specifically designed to capture this mixture of both long and short-term patterns in multivariate time series data whilst dealing with some of the scaling challenges that can be encountered with a neural network approach

Its model components are designed to address the different challenges
| Challenge | Approach |
| --- | --- |
| Short-term dependencies | Convolutional NN |
| Long-term dependencies | RNN |
| Data scaling | Autoregressive model |

LSTNet forumulation:
*  you have _n_ sets (or dimensions or variables) of data points observed over a particular time period. Based on this historical view, you want to predict the future data points up to a future point beyond the current time _T_, referred to as _h_ or the horizon.
*  predict the data point after the next time step after the horizon at time _T+1_. From this, you can create the input matrix _X_, which comprises all time series observations up to the current time _T_.

## LSTNet Architecture & Implementation
LSTNet is comprised of several components: a convolutional layer, recurrent and recurrent-skip layers, and an autoregressive model

### 1. Convolutional Layer
A Convolutional Neural Network or CNN is a type of neural network, often applied for the purposes of computer vision tasks such as image recognition, object detection and classification
* It can take an image as a numerical array (e.g. 28x28 pixels x 3 color channels RGB) input and apply a number of operations via a series of convolutional layers plus filters to extract features and ultimately feed into a decision which could be a categorical output to predict what the image might be.
* CNNs are not solely used for image processing, however in the LSTNet model, the CNN is used for feature extraction of local dependencies.
* In LSTNet, the convolutional layer is used to identify local dependencies
  * Extracts short-term patterns in the time dimension as well as local dependencies between variables

LSTNet architecture starts off with the multivariate time series, which is a number of different time series data, potentially of varying lengths
* These time series feed into the convolution layer
* This layer is used to identify the local dependencies which are subsequently fed into the recurrent layer
* The width of the kernel will account for all of the sequences

CNN with a ReLU function is used
* Multiple filters are used to process portions of the data at a time. These are two dimensional arrays of a width, w, multiplied by the height. Height is always set to the number of input variables
* Multiple filters or width and height: h<sub>k</sub> = ReLU(W<sub>k</sub> * X + b<sub>k</sub>
* To calculate the k-th output h<sub>k</sub>, you apply a ReLU function to the convolutional operation (which is the input matrix X multiplied by a set of weights W plus a bias value). This ReLU function will output either zero or the value of the convolutional operation. Whichever is greater.


### 2. Recurrent Layer
Recurrent layer is used to solve for the vanishing gradient
* RNN component is GRU layer with ReLU activation to overcome vanishing gradient
* When you get to the recurrent layer, the output from the convolutional layer is simultaneously fed to both the recurrent and the recurrent-skip layers.

The hidden state of recurrent units at time _t_ is computed as follows:
1. A reset gate r<sub>t</sub> which is used to decide how much of the past information to forget.
2. An update gate u<sub>t</sub> which is used to decide how much of the past information to be used in the future.
3. A current memory, c<sub>t</sub> which applies a ReLU function to use of the reset gate to filter out unnecessary data.
4. And, a final memory at the current time step
* The output of this layer is the hidden state at each time step.

Recurrent-skip layer is an element of the LSTNet model that looks for longer-term dependencies over a particular lag of time
* Purpose of the recurrent skip layer is to overcome some of the vanishing gradient issues that can come with very long-term correlations
* With this layer, it is easier to visualize time series data sets with clear patterns such as 60 minutes in an hour, 24 hours in a day, etc.

A dense layer combines the output from recurrent and recurrent-skip layers
* These two different outputs are then combined, which effectively creates a dense layer comprised of both outputs. This layer combines the hidden state of the recurrent layer and the _p_ hidden states of the recurrent skip layer.

In case of non-seasonal data, skip step _p_ is not useful
* In such cases, the temporal attention layer is used, which learns the weighted combination of hidden representations at each window of the input matrix
* Recurrent-skip layer incorporates a predefined hyperparameter p. However, this is not very helpful in the case of non-seasonal data or data with variations in periods. The temporal attention layer intends to overcome this limitation by learning the weighted combination of hidden representations at each window position of the input matrix.
* The output combines the weighted context vector (c<sub>t</sub>) with the last window hidden representation plus a linear projection operation, _b_

### 3. Autoregressive Model
Autoregressive model is used as a linear compenent to scale the output
* One major drawback of using a neural network model is that the scale of the outputs is not sensitive to the scale of the inputs. This means that when you have periodic datasets with peaks at different times, the scale is constantly changing which can lower the accuracy of the model.
* For the LSTNet model, a classical Autoregressive model is used as a linear component to scale the output based on the size of the input window. This allows the model to better address data with variable peaks
* Autoregressive model is a classical time series forecasting model to create forecasting results
  * Takes a set of observations on the previous time series and inputs these into a regression question to predict the next step

Final output combines the NN model and the AR component
