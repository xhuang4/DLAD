library(glmnet)
library(caret)
 
#read in features extracted by VGG16 model
ADCN.T1 <- read.table('./ADNI_AD_CN_T1_outputFeatures.txt',sep=" ",header=F)
ADCN.LJ <- read.table('./ADNI_AD_CN_LJ_outputFeatures.txt',sep=" ",header=F)
ADCN.T1.label <- read.table('./ADNI_AD_CN_T1_labels.txt',sep=" ",header=F)$V1
ADCN.LJ.label <- read.table('./ADNI_AD_CN_LJ_labels.txt',sep=" ",header=F)$V1
ADCN <- data.frame(rbind(ADCN.T1,ADCN.LJ))
ADCN <- ADCN[,apply(ADCN,2,function(x) quantile(x,0.95)!=0)]
ADCN.label <- append(ADCN.T1.label,ADCN.LJ.label)

MCI.T1 <- read.table('./ADNI_MCI_T1_outputFeatures.txt',sep=" ",header=F)
MCI.LJ <- read.table('./ADNI_MCI_LJ_outputFeatures.txt',sep=" ",header=F)
MCI.label <- read.table('./MCI_lables.txt',sep='\t',header=T) #with ID and pMCI/sMCI labels
MCI <- data.frame(rbind(MCI.T1,MCI.LJ))
MCI <- MCI[,colnames(ADCN)]
 
##penalized logistic regression to generate AD_DL_Score for MCIs
fit<- cv.glmnet(as.matrix(ADCN), ADCN.label, type.measure="auc",family="binomial")
pred.res <- predict(fit,as.matrix(MCI),s = c("lambda.min"),type="response")
pred.mci <- data.frame(MCI.label$PTID,MCI.label$TrueLabel, pred.res[,1])
colnames(pred.mci)[3] <- "AD_DL_Score"
write.table(pred.mci,'./MCI_AD_DL_Score.csv',sep=',',col.names = T,row.names = F)
 
#end

 
 