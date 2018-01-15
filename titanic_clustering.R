data<- read.csv('./data/train.csv',
                header = T,
                na.strings=c(""),
                stringsAsFactors = F)
data[is.na(data$Age),]$Age<-mean(data$Age, na.rm = T)

data[is.na(data$Embarked),]$Embarked<-'N'

data[data$Sex=='female',]$Sex <- 1
data[data$Sex=='male',]$Sex <- 2

data<- data[,c(-1,-4,-9,-11)]

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

train <- na.omit(train)
test <- na.omit(test)

survivedCluster <- kmeans(na.omit(train[,c(1:5)]), 2)
