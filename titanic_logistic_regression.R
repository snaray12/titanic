data<- read.csv('./data/train.csv',
                header = T,
                na.strings=c(""),
                stringsAsFactors = F)
data[is.na(data$Age),]$Age<-mean(data$Age, na.rm = T)

data[is.na(data$Embarked),]$Embarked<-'N'

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

null.model<- glm(Survived~1,
                 family=binomial(link = "logit"),
                 data=train)
full.model <- glm(Survived~.,
                  family=binomial(link = "logit"),
                  data=train)
forward.model <- step(null.model,
                      scope=list(lower=formula(null.model),
                                 upper=formula(full.model)),
                      direction = "forward")
backward.model <- step(full.model, trace = 0)
bothways.model <- step(null.model,
                       scope=list(lower=formula(null.model),
                                  upper=formula(full.model)),
                       direction = "both", trace=0)
summary(bothways.model)
anova(bothways.model, test = "Chisq")

library(pscl)
pR2(bothways.model)

fitted.results <- predict(bothways.model, newdata = test, type = 'response')
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
