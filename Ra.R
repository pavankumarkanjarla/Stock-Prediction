
library(readxl)       #load the readxl package
#install.packages('tidytext')
library(tidytext)     #load the tidytext package
#install.packages('Metrics')
library(Metrics)    #load the metrics library
#install.packages('dplyr')
library(dplyr)        #load the dplyr package

#install.packages('textdata')
library(textdata)    #load the textdata package
library(ggplot2)     #load the ggplot2 package

#stock data
df=read.csv("C:\\Users\\pavan\\Downloads\\AAPL (1).csv",header=TRUE,row.names='Date')
df    #load the stock data 

#Prophet Model (Additive regression model)
library(car)   #load the car package

ds=df$Date     #create a variable for the Date column from the stock data
df2 = data.frame(ds,y = as.numeric(df[,'Close'])) 
           #create a dataframe with the data variable and the closing value variable
df2        #output the dataframe

#install.packages("prophet")
library(prophet)       #load the Prophet package
prediction = prophet(df2)  #import the prophet model on the data
m1 = make_future_dataframe(prediction, periods = 365)  
                       #forecast the stock value for one year(365 days)
stockprice = predict(prediction,m1)  
                          #predict the future stock value for the prophet model
plot(prediction,stockprice)   #plot the predicted stock values
df3= data.frame(stockprice)   #create a dataframe with the predicted values 

#verification of assumptions of the Prophet Model
res= df3$yhat-df2$y #calculate the residuals of the fitted model
residuals=res[1:1259]     #output the residuals for the number of rows in the data
#durbin watson test 
durbinWatsonTest(res)  #conduct the durbin watson test on the residuals

#plotting the graph of residuals and fitted value
fittedvalues= df2$y    # assign a variable for the fitted values
plot(fittedvalues,residuals)   #plot the residuals and fitted values graph
abline(0,0)     #plot the line at 0,0
abline(lm(residuals~fittedvalues),col="red")  
               #plot the line of best to verify the linearity assumption

#Normal QQ plot for the residuals
qqnorm(residuals)    #plot the normal plot for the residuals of the model
qqline(residuals)    #add the normal line to the plot
hist(residuals)      #plotting the histogram of the residuals

#dividing the data into training and testing
dim(df)          #output the dimensions of the data
train=df[1:1177,]   #divide the data into training set
head(train)      #output the top six values of the training set
tail(train)      #output the last six values of the training set
test=df[1178:1259,]  #divide the data into the testing set
tail(test)        #output the last six values of the testing set 
head(test)        #output the first six values of the testing set

#time series forecasting models
#arima
df   #output the dataframe

#install.packages("rugarch")
#install.packages("parallel")
#install.packages("forecast")

library(forecast)   #load the forecast library
library(rugarch)    #load the rugarch library
library(tseries)    #load the tseries library

df1=data.frame(train[,'Close'])
               #create a new dataframe with the closing value of the stock
df1
x = ts(df1, frequency = 1)   
       #create a time series with a frequency for every day  
print(adf.test(x))  #output the ADF value of the close variable time series
plot(x)     #plot the timeseries of the close variable

train1=train[,5]  #create a training set with prediction variable
train1     #output the training model
test1=test[,5]  #create a testing set with the prediction variable
test1      #output the testing model

# The time series is not stationary
#Therefore we need to perform the first order logarithmic difference 
#to make the time series stationary
stock = diff(log(train1),lag=1)  
                   #perform the first order logarithmic difference on training set
stock    #output the difference value
stock1 = diff(log(test1),lag=1)   
                 #perform the first order logarithmic difference on testing set
stock1    #output the difference value
plot(stock,type ='l',main ='log return plot')  #plot the difference values
y = ts(stock, frequency = 1)  #create the time series for the difference value
plot(y)   #plot the time series 

#ADF test 
print(adf.test(y, alternative = c("stationary", "explosive"),k = 0))
      #perform the ADF test on the stationary time series

acf(y)      #plot the ACF graph for the stationary time series
pacf(y)     #plot the PACF graph for the stationary time series

fit=arima(stock, c(0, 1, 1),seasonal = list(order = c(0, 1, 1), period = 12))
   #train the arima model on the data
pred=predict(fit, n.ahead = 81)  #forecast the stock values for a period of 81 days
pred$pred   #output the predicted values

ts.plot(2.718^stock1,2.718^pred$pred,log='y',col=c("red","blue"),union=TRUE)
 #plot the time series of the predicted and the actual values
legend(x="topleft",legend=c("Actual","Predicted"),fill = c("red","blue"))
     #fit the legend in the plot
mape(stock1,pred$pred)   
          #calculate the Mean Percentage absolute error of the model prediction
accuracy(stock1,pred$pred)  #calculate the performance metrics of the model

#shuffling the rows of the training data
trains = train[sample(1:nrow(train)),] 
                #shuffle the rows of the data to train the model
head(trains)    #output the first six rows of the shuffled data

#smoothing method
#install.packages('fpp2')
library(fpp2)        #load the fpp2 library
library(tidyverse)   #load the tidyverse library

#Exponential smoothing method

sm=ses(train1,alpha=0.2,h=82) #fir the SES model into the training data
autoplot(sm)    #plot the ses model
ts.plot(df$Close)  #plot the time series of the close variable
sm   #output the  model configuration
summary(sm)    #output the summary of the model
sm=data.frame(sm)  #create a dataframe for the model prediction
sm                 #output the dataframe
mape(sm$Point.Forecast,test1)   #calculate the error in the predicted values

#exponential smoothing using the difference method
dif=diff(train1)   #calculate the first order difference for the Close variable
sesdif = ses(dif, alpha = .2, h = 82) #fit the ses model into the difference data
autoplot(sesdif)   #plot the ses model
sesdif=data.frame(sesdif)  #create a dataframe of the predicted values
diftest=diff(test1)  #calculate the first order difference for the testing data 
mape(sesdif$Point.Forecast,diftest)  #calculate the error in the predicted values

#holt smoothing method
holt = holt(train1,seasonal="multiplicative",h=82,level=0.95)
     #train the holt model on the data for the confidence interval of 95%
holt.pred=predict(holt, prediction.interval = TRUE)
     #forecast the stock values using the holt model
holt.pred   #output the predicted values 
autoplot(holt.pred)   #plot the predicted values of the stock
holt.pred=data.frame(holt.pred)  #create a dataframe with the predicted values
mape(holt.pred$Point.Forecast,test1)  #calculate the error in the predicted values

#TBATS forecasting model
tbats=tbats(train1)  #train the TBATS model on the training data
summary(tbats)    #output the summary of the model
autoplot(tbats)   #plot the model
forecast=forecast(tbats, h = 82)   
                #forecast the stock values using the TBATS model
autoplot(forecast)  #plot the predicted values 
forecasttbats=data.frame(forecast)  #create a dataframe with the predicted values
mape(test1,forecasttbats$Point.Forecast)   
                                   #calculate the error in the predicted values

#Machine Learning Models
#Decision tree

#install.packages('caret')
library(caret)         #load the caret library
library(quantmod)      #load the quantmod library
library(lubridate)     #load the lubridate library
df        #output the dataframe
df3=df[,c("Open","High","Low","Volume","Close")]   
                      #create a new dataframe with the predictor columns
df3            #output the new dataframe

train2=df3[1:1177,]    #create the training set in the new dataframe
test2=df3[1178:1259,1:4]   #create the testing set in the new dataframe
train2s=train2[sample(1:nrow(train2)),]   #shuffle the training rows
train2s       #output the shuffled rows

#install.packages('party')
library(caTools)   #load the Catools library
library(party)     #load the party library
library(dplyr)     #load the dplyr library
library(magrittr)  #load the magrittr library
train2    #output the training data
test2     #output the testing data
#install.packages('rpart')
#install.packages('rpart.plot')
library(rpart)     #load the rpart library
library(rpart.plot) #load the rpart.plot library
tree=rpart(Close~Open+High+Low+Volume,data=train2s,method = 'anova') 
    #train the decision tree model on the training data
tree  #output the trained model
p1=predict(tree,test2,method='anova')   
                   #predict the stock values using this decision tree model
p1                 #output the predicted values
test[,5]           #output the true values of the stock
accuracy(p1,test[,5]) #calculate the performance statistics of the predicted values

#Leave One Out Cross Validation
set.seed(5)   #set the seed vale
control=trainControl(method ="LOOCV")  #train the loose one out cross validation
tree1=train(Close~Open+High+Low+Volume,data=train2,method='rpart',trControl=control)
       #implement the cross validation for the decision tree model
print(tree1)   #output the validated tree model
plot(tree1)    #plot the validated tree
p10=predict(tree,test2)   #predicted the stock values using this validated model
p10    #output the predicted values

#Random Forest Model
library(randomForest)   #load the random forest library
rf=randomForest(Close~., data=train2, proximity=TRUE)  
                       #train the random forest model on the training data
print(rf)     #output the model
p2=predict(rf, test2)   #predict the stock values of the data
p2     #output the predicted values
resultrf=data.frame(actual=df$Close[1178:1259],prediction=p2) 
                 #create a dataframe with the actual and the predicted values
resultrf         #output the created dataframe
ts.plot(resultrf,col=c("red","blue"),xlab="Days",ylab="Stock Value") 
                  #plot the timeseries for the predicted values and true values
legend(x="topleft",legend=c("Actual","Predicted"),fill = c("red","blue"))
                  #add legend to the plot
mape(resultrf$prediction,resultrf$actual)   
                  #calculate the error in the predicted and the true values


#Neural Network (ANN)

scaleddata=scale(df)   #scale the dataframe
#create a function to normalize the data
normalize=function(a) {
  return ((a - min(a)) / (max(a) - min(a)))
}     
normdata=data.frame(lapply(df, normalize))  
                     #apply the normalize function to the data
normdata       #output the normalized data

nntrain=normdata[1:1177,]    #create a training set in the normalized data
nntest=normdata[1178:1259,]  #create a testing set in the normalized data

#Neural Network
library(neuralnet)   #load the neural net library
nn=neuralnet(Close~Open+High+Low+Volume,data=nntrain,hidden=c(7,4),linear.output=TRUE,threshold = 0.01)
         #train the neural network on the training data with the hidden layers
nn       #output the neural net model
plot(nn)  #plot the neural network
nn$result.matrix  #output the resukt matrix of the neural network
nnpred=nntest[,c("Open","High","Low","Volume")]  
                #create a dataframe with the predictor variables on the testing set
nnpred       #output the dataframe
nnresults=neuralnet::compute(nn,nnpred)    #predict the stock values 
nnresults   #output the results
results=data.frame(actual = nntest$Close, prediction = nnresults$net.result)
 #create a dataframe with the predicted and the actual stock values
results    #output the result dataframe
mape(results$actual,results$prediction)  
                     #calculate the error in the predicted values

ts.plot(results,col=c("red","brown"),xlab="Days",ylab="Stock Value")
#plot the time series of the predicted and actual stock values
legend(x="topleft",legend=c("Actual","Predicted"),fill = c("red","brown"))
#add legend to the plot
