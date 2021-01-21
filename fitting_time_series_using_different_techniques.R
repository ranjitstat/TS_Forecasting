

#######################Fitting of GRNN#########################

library(yager)

mydata<-read.csv("D:\\Drive\\NCIPM\\Tomato_November_2020\\LM.csv",header=T)

attach(mydata)
nrow(mydata)
names(mydata)

train<-mydata[1:69,]
test<-mydata[70:79,]
X=as.matrix(train[,2:9])
Y=train[,1]

X_pred=as.matrix(test[, 2:9])
gnet <- grnn.fit(x = X, y = Y)

predicted<-grnn.parpred(gnet, X_pred)
data.frame(predicted)

library(MLmetrics)

MAPE(predicted,test[,1])
MSE(predicted,test[,1])
########################Fitting of GBM##################################
mydata<-read.csv("D:\\Drive\\NCIPM\\Tomato_November_2020\\TH.csv",header=T)

attach(mydata)
nrow(mydata)
names(mydata)

train<-mydata[1:50,]
test<-mydata[51:60,]

library(gbm)

gbm1 <- gbm(train[,1] ~ ., data = train)  
print(gbm1)
summary(gbm1)
best.iter <- gbm.perf(gbm1, method = "cv")
print(best.iter)
predicted <- predict(gbm1, newdata = test, n.trees = 10, type = "link")
MAPE(predicted,test[,1])
MSE(predicted,test[,1])

####################Fitting of KNN#########################

library(FNN)

mydata<-read.csv("D:\\Drive\\NCIPM\\Tomato_November_2020\\TH.csv",header=T)
attach(mydata)
nrow(mydata)
names(mydata)
train<-mydata[1:50,]
test<-mydata[51:60,]

knn<-knn.reg(train, test = test, train[,1], k = 2, algorithm=c("kd_tree", 
                                                               "cover_tree", "brute"))
summary(knn)

predicted<-(knn$pred)

MAPE(predicted,test[,1])
MSE(predicted,test[,1])

###########Fitting of random forest#######################
set.seed(100)

library(randomForest)

mydata<-read.csv("D:\\Drive\\NCIPM\\Tomato_November_2020\\TH.csv",header=T)
attach(mydata)
nrow(mydata)
names(mydata)
train<-mydata[1:50,]
test<-mydata[51:60,]

rf = randomForest(train[,1] ~ ., data = train, importance=TRUE,proximity=TRUE)

print(rf)
summary(rf)
importance(rf)
varImportance(rf)
plot(rf)

predictions = predict(rf, newdata = test)
data.frame(predictions)

library(MLmetrics)
MAPE(predictions,test[,1])
MSE(predictions,test[,1])


round(importance(rf), 2)



set.seed(647)
myiris <- cbind(iris[1:4], matrix(runif(96 * nrow(iris)), nrow(iris), 96))
result <- rfcv(myiris, iris$Species, cv.fold=3)
plot(n.var, error.cv, log="x", type="o", lwd=2)

result <- replicate(5, rfcv(myiris, iris$Species), simplify=FALSE)
error.cv <- sapply(result, "[[", "error.cv")
matplot(result[[1]]$n.var, cbind(rowMeans(error.cv), error.cv), type="l",
        lwd=c(2, rep(1, ncol(error.cv))), col=1, lty=1, log="x",
        xlab="Number of variables", ylab="CV Error")


getTree(rf, k=1, labelVar=FALSE)

print(rf)
predictions = predict(rf, newdata = test)
data.frame(predictions)

library(MLmetrics)
MAPE(test[,1], predictions)

predictions = predict(rf, newdata = test)
mape(test$Sales, predictions) 
varImpPlot(rf)

set.seed(100)
rf_revised = randomForest(Sales ~ Inventory + year + yday  + weekdays + weekend, data = train)
print(rf_revised) 

predictions = predict(rf_revised, newdata = train)
mape(train$Sales, predictions) 


predictions = predict(rf_revised, newdata = test)
mape(test$Sales, predictions)

###################Fitting of automatic machine learning######################
Library(h2o)
mydata<-read.csv("C:\\Users\\Ranjit\\Desktop\\RANJIT\\Debopam\\LM.csv",header=T)
attach(mydata)
nrow(mydata)
names(mydata)
train<-mydata[1:60,]
test<-mydata[61:79,]
h2o.init()



mydata <- h2o.importFile("C:\\Users\\Ranjit\\Desktop\\RANJIT\\Debopam\\LM_1.csv",header=T)


data_split <- h2o.splitFrame(data = mydata, ratios = 0.8, seed = 1234)
train <- data_split[[1]]
valid <- data_split[[2]]

predictors <- c("x1", "x2", "x3", "x4", "x5")
response <- "y"
h2o.gbm(x = predictors, y = response, training_frame = train,
        validation_frame = valid, seed = 1234)

aml<- h2o.automl(
  x = predictors, y = response, training_frame = train,
  validation_frame = valid, seed = 1234,  nfolds = 5,
  sort_metric = c("AUTO", "deviance", "logloss", "MSE", "RMSE", "MAE", "RMSLE", "AUC",
                  "AUCPR", "mean_per_class_error"),
  
)
exa <- h2o.explain(aml, valid)



lb <- h2o.get_leaderboard(aml)
head(lb)
library(tibble)
mod_ids <- as_tibble(aml@leaderboard$model_id)


for(i in 1:nrow(mod_ids)) {
  
  aml1 <- h2o.getModel(aml@leaderboard[i, 1]) # get model object in environment
  h2o.saveModel(object = aml1, "C:\\Users\\Ranjit\\Desktop\\RANJIT\\Debopam") # pass that model object to h2o.saveModel as an argument
  
}


summary(aml)
cars <- h2o.importFile("https://s3.amazonaws.com/h2o-public-test-data/smalldata/junit/cars_20mpg.csv")

# convert response column to a factor
cars["economy_20mpg"] <- as.factor(cars["economy_20mpg"])

# set the predictor names and the response column name
predictors <- c("displacement", "power", "weight", "acceleration", "year")
response <- "economy_20mpg"

# split into train and validation sets
cars_split <- h2o.splitFrame(data = cars, ratios = 0.8, seed = 1234)
train <- cars_split[[1]]
valid <- cars_split[[2]]

# try using the `validation_frame` parameter:
# train your model, where you specify your 'x' predictors, your 'y' the response column
# training_frame and validation_frame
cars_gbm <- h2o.gbm(x = predictors, y = response, training_frame = train,
                    validation_frame = valid, seed = 1234)

# print the auc for your model
print(h2o.auc(cars_gbm, valid = TRUE))
