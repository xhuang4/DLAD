library(SuperLearner)
library(pROC)
library(ggcorrplot)
library(xgboost)
library(survival)
library(glmnet)

#load all biomarker data
MCIADscore <- read.table('./MCI_AD_DL_Score.csv',sep=',',header=T)
data.all <- read.table('./ADNIMERGE_20190819.csv', sep=',',header = T) #download from ADNI database

#select MCI patients and data columns
data.all <- data.all[,c("PTID","COLPROT","VISCODE","AGE","PTGENDER","PTEDUCAT","APOE4","CDRSB","ADAS13","MMSE","FAQ")]
data.all <- merge(data.all[data.all$VISCODE=="bl",],MCIADscore,by="PTID")
train.data <- na.omit(data.all[data.all$COLPROT=="ADNI1",])
test.data <- na.omit(data.all[data.all$COLPROT=="ADNI2"|data.all$COLPROT=="ADNIGO",])

#Ensemble Learning
SL.library <- c("SL.glmnet", "SL.rpart","SL.xgboost","SL.gam")
cvControl <- SuperLearner.CV.control(V=10) #10 fold cross validation
seed <- 12345  

set.seed(seed) #demographic+genetic
model1 <-SuperLearner(train.data$TrueLabel,train.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT")],family = binomial(),SL.library = SL.library,cvControl = cvControl,method = "method.AUC")
pred1 <- predict(model1,test.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT")])
roc(test.data$TrueLabel,pred1$pred[,1])

set.seed(seed) #demographic+genetic+image
model2 <- SuperLearner(train.data$TrueLabel,train.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","AD_DL_Score")],family = binomial(),SL.library = SL.library,cvControl = cvControl,method = "method.AUC")
pred2 <- predict(model2,test.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","AD_DL_Score")])
roc(test.data$TrueLabel,pred2$pred[,1])

set.seed(seed) #demographic+genetic+cognitive
model3 <- SuperLearner(train.data$TrueLabel,train.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13")],family = binomial(),SL.library = SL.library,cvControl = cvControl,method = "method.AUC")
pred3 <- predict(model3,test.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13")])
roc(test.data$TrueLabel,pred3$pred[,1])

set.seed(seed) #demographic+genetic+cognitive+image
model4 <- SuperLearner(train.data$TrueLabel,train.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13","AD_DL_Score")],family = binomial(),SL.library = SL.library,cvControl = cvControl,method = "method.AUC")
pred4 <- predict(model4,test.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13","AD_DL_Score")])
roc(test.data$TrueLabel,pred4$pred[,1])
 
#XGBoost Importance plot  
temp.obj <- model4$fitLibrary$SL.xgboost_All$object
importance_matrix <- xgb.importance(model = temp.obj)
xgb.ggplot.importance(importance_matrix,n_clusters = 1)

#Penalized Cox model 
train.data$PTGENDER <- (train.data$PTGENDER=="Female")*1
test.data$PTGENDER <- (test.data$PTGENDER=="Female")*1
coxfit <- cv.glmnet(as.matrix(train.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13","AD_DL_Score")]), Surv(wkdata1$VISCODE,wkdata1$TrueLabel), type.measure="deviance",family="cox")
pred.res <- predict(coxfit,as.matrix(test.data[,c("AGE","APOE4","PTGENDER","PTEDUCAT","CDRSB","FAQ","MMSE","ADAS13","AD_DL_Score")]),s = c("lambda.min"),type="response")
roc(test.data$TrueLabel,pred.res[,1])
 

 