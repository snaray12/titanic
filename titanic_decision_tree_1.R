data<- read.csv('./data/train.csv',
                header = T,
                na.strings=c(""),
                stringsAsFactors = F)

data[is.na(data$Embarked),]$Embarked<-'S'

library(mice)

tempData <- mice(data,m=5,maxit=50,meth='pmm',seed=500)

data <- complete(tempData,1)


#data2<-within(data,
#       Name<-data.frame(do.call('rbind',
#                               strsplit(as.character(Name),
#                                        ',',
#                                        fixed=TRUE))))

#data2<-transform(data,
#          lapply({l<-list(Name);names(l)=c('Name');l},
#                 function(x)do.call(rbind,
#                                    strsplit(as.character(x), ',', fixed=TRUE))), stringsAsFactors=F)
#data2<-transform(data2,
#                 lapply({l<-list(Name.2);names(l)=c('Name.2');l},
#                        function(x)do.call(rbind,
#                                           strsplit(as.character(x), '.', fixed=TRUE))), stringsAsFactors=F)

data$Title<-str_extract_all(data$Name, "[A-Za-z]+\\.")

data<- data[,c(-1,-4,-9,-11)]

prop.table(table(data$Survived))
prop.table(table(data$Pclass))
prop.table(table(data$Sex))
prop.table(table(data$SibSp))
prop.table(table(data$Parch))
prop.table(table(data$Sex, data$Survived))
prop.table(table(data$Pclass, data$Survived))
prop.table(table(data$SibSp, data$Survived))
prop.table(table(data$Parch, data$Survived))

data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.factor(data$Sex)
data$SibSp <- as.factor(data$SibSp)
data$Parch <- as.factor(data$Parch)
data$Embarked <- as.factor(data$Embarked)

set.seed(1234)
indexes = sample(1:nrow(data), size=0.2*nrow(data))
test<- data[indexes,]
train<- data[-indexes,]


library(rpart)
fit <- rpart(Survived~., data=train, method="class")

anova(fit, test = "Chisq")

library(pscl)
pR2(fit)

fitted.results <- predict(fit, newdata = test, type = 'prob')
fitted.results <- ifelse(fitted.results>0.5,1,0)
misClassificationError <- mean(fitted.results!=test$Survived)
print(paste('Accuracy ', 1-misClassificationError))

library(ROCR)

pr <- prediction(fitted.results, test$Survived)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[1]
print(paste("Area under curve ", auc))
