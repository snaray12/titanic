Math.cbrt <- function(x) {
  sign(x) * abs(x)^(1/3)
}

data<- read.csv('./data/train.csv',
                header = T,
                na.strings=c(""),
                stringsAsFactors = F)

data.test<- read.csv('./data/test.csv',
                     header = T,
                     na.strings=c(""),
                     stringsAsFactors = F)
aggregate(Fare~Parch+Embarked+Pclass, data[grep("B",data$Cabin),], mean)

data[is.na(data$Embarked),]$Embarked <- 'S'

library(mice)
md.pattern(data)

library(VIM)

aggr_plot <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

tempData <- mice(data,m=5,maxit=50,meth='pmm',seed=500)

summary(tempData)

completedData <- complete(tempData,1)

xyplot(tempData,Survived ~ Age+Embarked,pch=18,cex=1)

densityplot(tempData)

stripplot(tempData, pch = 20, cex = 1.2)

anyNA(completedData)

colSums(is.na(completedData))


dim(data)

apply(data, 2, function(x) length(unique(x)))

apply(data.test, 2, function(x) length(unique(x)))

colSums(is.na(data))

data[is.na(data$Age),]$Age<-mean(data$Age, na.rm = T)

data.test[is.na(data.test$Age),]$Age<-mean(data.test$Age, na.rm = T)

data.test[data.test[,"Parch"]==9,]$Parch <- 6

# Sturge’s Rule
k1 <- 1+(3.22 * log(length(data$Age)))

print(paste("Number of class intervals (bins)", k1))

## Doane’s Rule

## Scott’s Rule
k3 <- 3.49*sd(data$Age)*length(data$Age)^-(1/3)

print(paste("Number of class intervals (bins)", k3))

## Rice’s Rule
k4 <- Math.cbrt(length(data$Age))*2
print(paste("Number of class intervals (bins)", k4))

## Freedman-Diaconis’s Rule
k5 <- 2* IQR(data$Age)*(length(data$Age))^(-1/3)
print(paste("Number of class intervals (bins)", k5))

## Age range setting
# 0-5 Infant - 1
# 5-16 Child - 2
# 16 - 32 Youth - 3
# 32 - 48 Adult - 4
# 48 - 64 Middle Aged - 5
# 64 - 80 Senior - 6
data$AgeRange <- 6

data[data$Age<=5,]$AgeRange <- 1
data[(data$Age>5 & data$Age <=16),]$AgeRange <- 2
data[(data$Age>16 & data$Age <=32),]$AgeRange <- 3
data[(data$Age>32 & data$Age <=48),]$AgeRange <- 4
data[(data$Age>48 & data$Age <=64),]$AgeRange <- 5

data.test$AgeRange <- 6

data.test[data.test$Age<=5,]$AgeRange <- 1
data.test[(data.test$Age>5 & data.test$Age <=16),]$AgeRange <- 2
data.test[(data.test$Age>16 & data.test$Age <=32),]$AgeRange <- 3
data.test[(data.test$Age>32 & data.test$Age <=48),]$AgeRange <- 4
data.test[(data.test$Age>48 & data.test$Age <=64),]$AgeRange <- 5
#data[data$Age>64,]$AgeRange <- 6

data.test[is.na(data.test$Fare),]$Fare<-mean(data.test$Fare,na.rm=T)


data<- data[,c(-1,-4, -6, -9,-11)]

#data.test<-data.test[,c(-1,-3,-8,-10)]

prop.table(table(data$Survived))
prop.table(table(data$Pclass))
prop.table(table(data$Sex))
prop.table(table(data$SibSp))
prop.table(table(data$Parch))
prop.table(table(data$Sex, data$Survived))
prop.table(table(data$Pclass, data$Survived))
prop.table(table(data$SibSp, data$Survived))
prop.table(table(data$Parch, data$Survived))

factor.cols <- c("Pclass", "Sex", "SibSp", "Parch", "Embarked", "AgeRange")

for(fc in factor.cols) {
  data[,fc]<-as.factor(data[,fc])
  data.test[,fc]<-as.factor(data.test[,fc])
}

levels(data.test$Embarked)<-levels(data$Embarked)

#data$Survived <- as.factor(data$Survived)
#data$Pclass <- as.factor(data$Pclass)
#data$Sex <- as.factor(data$Sex)
#data$SibSp <- as.factor(data$SibSp)
#data$Parch <- as.factor(data$Parch)
#data$Embarked <- as.factor(data$Embarked)

set.seed(1234)
indexes = sample(1:nrow(data), size=0.2*nrow(data))
test<- data[indexes,]
train<- data[-indexes,]

library(randomForest)

fit <- randomForest(as.factor(Survived)~., data=train, ntree=30, mtry=2)
summary(fit)

predicted <- predict(fit, newdata = test, type = 'class')

t<- table(predictions=predicted, actual=test$Survived)

t

## Accuracy metric
sum(diag(t))/sum(t)

varImpPlot(fit)

#Plotting the ROC curve and calculating AUC metric

library(pROC)

predictionsWithProb <- predict(fit, test, type="prob")

auc <- auc(test$Survived, predictionsWithProb[,2])

print(paste("AUC:",auc))

plot(roc(test$Survived, predictionsWithProb[,2]))

## To find the best mtry

bestmtry <- tuneRF(train, train$Survived,
                   ntreeTry =30, stepFactor = 1.2,
                   improve = 0.01, trace = T, plot = T)

bestmtry

prop.table(table(predicted, test$Survived))

accuracies <- c()

accuracy <- predicted == test$Survived

accuracies <- c(accuracies, accuracy)

print(paste("Accuracy", sum(accuracies)/nrow(test)))


data.test$predicted <- predict(fit, newdata = data.test, type = 'class')

gender_submission<-data.test[,c(1,13)]

colnames(gender_submission)<-c("PassengerId", "Survived")

write.csv(gender_submission, "./data/gender.submission.csv", row.names = F)
