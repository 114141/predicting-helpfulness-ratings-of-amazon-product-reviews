#uncomment to install and load packages used in this process
#install.packages("rjson") 
#install.packages("jsonlite")
#insall.packages("tidyr")
#install.packages("data.table")
#install.packages("stringr")
#install.packages("psych")
#install.packages("ggplot2")
#install.packages("corrplot")
#install.packages("plyr")
#install.packages("tm")
#install.packages("tidytext")
#install.packages("tidyverse")
#install.packages("SnowballC")
#install.packages("dplyr")
#install.packages("wordcloud")
#install.packages("class")
#install.packages("e1071")
#install.packages("stats")
#install.packages("resample")
#install.packages("MASS")
#install.packages("devtools")
#install_github("factoextra") 
#install.packages("nortest")
library(stringr)
library(tidyr)
library(rjson)
library(jsonlite)
library(data.table)
library(psych)
library(ggplot2)
library(corrplot)
library(tm)
library(plyr)
library(tidytext)
library(tidyverse)
library(SnowballC)
library(dplyr)
library(wordcloud)
library(class)
library(e1071)
library(stats)
library(resample)
library(MASS)
library(devtools)
library(factoextra)
library(nortest)




#READING DATA INTO R

#original data set found at: "snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz"

#note: original dataset is not in strict json see file "convert_reviews.py" to turn the data set into strict json which was done
#initially and which created the file "new_reviews_beauty.json"

#uploading json new_reviews into data frame 
json_reviews_amazon <- jsonlite::stream_in(file("new_reviews_beauty.json"))

#check datatypes of attributes of review data
sapply(json_reviews_amazon, class)

#checking to examine first like (shows first "3" rows)
head(json_reviews_amazon, 3)

#CREATING NEW FEATURES

#subsetting the variables we are going to use in our model - "helpful", "reviewtext" and "overall"
newdf <- json_reviews_amazon[c(4, 6:7)]

#checking that we've subsetted correctly
head(newdf, 3)

#creating new features - "helpful numerator" and "helpful denominator"

#replace all ":" with "," in the helpful column
newdf$helpful = gsub(":",",",newdf$helpful)

#splitting the helpful column into numerator and denominator
newdf <- newdf %>% separate(col = helpful, into = c("helpful_num", "helpful_denom"), sep = ",")

#removing unwanted characters leftover like "c", "(" and ")" from respective columns
newdf$helpful_num = gsub("c","",newdf$helpful_num)
newdf$helpful_num = gsub("[(]","",newdf$helpful_num)
newdf$helpful_denom = gsub("[)]","",newdf$helpful_denom)


#checking if there are any missing values
missing <- newdf[!complete.cases(newdf), ]
#there are no mising values
missing

#check classes again for data types that we might need to change
sapply(newdf, class)

#since helpful_num and helpful_denom are char change them to numeric
newdf$helpful_num <- as.numeric(as.character(newdf$helpful_num))
newdf$helpful_denom <- as.numeric(as.character(newdf$helpful_denom))

#describe the dataset
describe(newdf)
#note that maximum numbers are very high above mean in standard deviations which indicates that there are outliers which could throw off model

#EXPLORATORY DATA ANALYIS AND CREATING A CLASS LABEL

#boxplot of distrubution of data
names = c("overall","num","denom")
newplot <- boxplot(newdf$overall, newdf$helpful_num, newdf$helpful_denom, names=names, las=3)
newplot

#we will only use records where the amount of raters (ie. denominator column) is more than 10 so that
#we don't artifically rate "helpful" reviews as non-helpful just because they are not rated by people.
#saving this to new dataframe which we will use going forward
ten_newdf <- newdf[which(newdf[,2]>10),]
ten_newdf

#check to missing values
missing_ten <- ten_newdf[!complete.cases(ten_newdf), ]
#there are no mising values
missing_ten

#new plot after taking on values more than 10 
names = c("overall","num","denom")
ten_newplot <- boxplot(ten_newdf$overall, ten_newdf$helpful_num, ten_newdf$helpful_denom, names=names, las=3)
ten_newplot

#adding a new column for word count
#nchar from stringr package
ten_newdf$reviewWordCount <- nchar(gsub('[^ ]+', '',ten_newdf$reviewText))+1

#now creating a new feature - ie. the class label for "helpful".
ishelpful = 0.6
ten_newdf$helpful <- ifelse((ten_newdf$helpful_num/ten_newdf$helpful_denom)>ishelpful, 1, 0)

#checking to see how imbalanced the dataset is. it's really impalanced towards "helpful" reviews
counthelpful <- count(ten_newdf, vars = helpful)
#0 - 7226
#1 - 52213
7226/52213 = 0.1383947
0.1383947 * 100
#unhelpful reviews makes up 13.83947% of data set
100 - 13.83947 
#helpful reviews makeup 86.16053% of data set
#perhaps we will downsample later or use a cost function in our models


#using corrplot() to visualize correlation
subsetforcor <- ten_newdf[c(1,2,4,5,6)]
corrplot(cor(subsetforcor), method="color")
#There is a small correlation between the 'helpful' rating and our 
# overall score - ie. the star rating that a review has given the product.

#plotting histograms of overall score
overall <- ten_newdf$overall
hist(overall)
#most ratings are positive - many are five stars.

#DISTRUBUTION OF ATTRIBUTES

#We cannot use Shapiro test because
#of how big our dataset is so we use
#Anderson-Darling normality test, which works 
#for larger sample sizes.

ad.test(ten_newdf$helpful_denom)
#p-value < 2.2e-16 so we reject null hypothesis 
#and no significant departure from normality was found

#The test rejects the hypothesis of normality when the p-value 
#is less than or equal to 0.05. If we fail the normality test we can
#with 95% confidence believe that the data does not fit the normal 
#distribution. 

ad.test(ten_newdf$helpful_num)
# p-value < 2.2e-16  #we reject null hypothesis 
#and no significant departure from normality was found


ad.test(ten_newdf$overall)
# p-value < 2.2e-16  #we reject null hypothesis 
#and no significant departure from normality was found


ad.test(ten_newdf$reviewWordCount)
# p-value < 2.2e-16  #we reject null hypothesis 
#and no significant departure from normality was found


##TEXT MINING

#cleaning text as a part of preprocess (using tm package)
reviews <- ten_newdf$reviewText

#reading text into review corpus for processing
review_corpus <- Corpus(VectorSource(reviews))

#change to lowercase
review_corpus <- tm_map(review_corpus, content_transformer(tolower))
#remove numbers
review_corpus <- tm_map(review_corpus, removeNumbers)
#remove punctuation
review_corpus <- tm_map(review_corpus, removePunctuation)
#remove stop words and "the" and "and"
review_corpus <- tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
#stemming words
review_corpus <- tm_map(review_corpus, stemDocument, language = "english")
#removing white space
review_corpus <- tm_map(review_corpus, stripWhitespace)

#checking out corpus final outcome
inspect(review_corpus[1])

#creating a document-term matrix: where documents 
#are rows, and the words are the columns. 
#frequency of the term in the document are the entries.
review_doctm <- DocumentTermMatrix(review_corpus)
review_doctm

#inspecting five rows of matrix
inspect(review_doctm[305:310, 305:310])


#the sparsity is very high. we will remove low frequency words
#as they are possibly typos or otherwise uninteresting
review_doctm = removeSparseTerms(review_doctm, 0.98)
review_doctm


#inspecting five rows of matrix again
inspect(review_doctm[305:310, 305:310])


#finding tf-idf which finds the relative importance of a word to a document.
reviewdoctm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
#this process is noting that there are empty reviews (ie. documents) 
#this is likely that documents have been dropped when we cleaned text 
#(i.e. removing stop words, etc)
reviewdoctm_tfidf = removeSparseTerms(reviewdoctm_tfidf, 0.98)
reviewdoctm_tfidf
#sparsity is now 94% which is suitable for our purposes

#checking the first document
inspect(reviewdoctm_tfidf[1,1:15])


#converting doctm_tfidf matrix to dataframe
tfidfam <- as.matrix(reviewdoctm_tfidf)
tfidfdf <- as.data.frame(tfidfam)
#checking out how many terms in the dataframe 
#(i.e. how many words we have)
ncol(tfidfdf)

#creating new dataframe for reviews and tf-idf to inspect before adding as feature to full dataset
n_reviews <- cbind(reviews, tfidf)
#view sample of new datafram
head(n_reviews)

#binding tfidfdf w working dataframe
combineddf <- cbind(ten_newdf, tfidfdf)

#checking class of new column
sapply(combineddf, class)

#ranking of terms
wordsdecre <- data.frame(sort(colSums(as.matrix(reviewdoctm_tfidf)), decreasing=TRUE))
wordsdecre #decreasing list of tfidf words
#now making wordcloud
wordcloud(rownames(wordsdecre), wordsdecre[,1], max.words=60, colors=brewer.pal(4, "Dark2"))
#word cloud shows that hair, product, skin, use are significant terms


#SPLITTING TRAIN AND TEST DATA SETS FOR MORE PROCESSING

trainrows <- sample(nrow(combineddf),nrow(combineddf)*0.80)
combineddf.train = combineddf[trainrows,]
combineddf.test = combineddf[-trainrows,]
#training on 80% of dataset, and testing on remaining 20%

#PCA ON TRAINING DATA

pcadata.train <- combineddf.train[c(2,4:5,7:551)] 
#including the variables: denominator, wordcount, overall, and all terms

#argument scale - the variable standard deviations (the scaling applied to each variable) ie. normalized
pca.train <- prcomp(pcadata.train , scale = T)

#checking prcomp attributes - sdev, rotation, center, x
names(pca.train)

#show summary
summary(pca.train)

#In the results of prcomp, you can see the principal components (pca.train$x), 
#the eigenvalues (pca.train$sdev) give info on the magnitude of 
#each principal component, and the loadings (pca.train$rotation). 

pca.train$sdev
length(pca.train$sdev)
pca.train$rotation
dim(pca.train$rotation)
pca.train$x
dim(pca.train$x)

#squaring the eigenvalues to the get variance explained by principal comps
plot(cumsum(pca.train$sdev^2/sum(pca.train$sdev^2)))


#FURTHER ANALYSIS OF PRINCIPAL COMPONENTS FOR FEATURE SELECTION

#extract the results for variables and individuals
get_pca(pca.train, element = c("var", "ind"))

#extracting the result of variables
var <- get_pca_var(pca.train)
var

#coordinates of variables
head(var$coord)

#visualization of the variables
fviz_pca_var(pca.train)
#they overlap a lot here, but skin, overall, hair, product jump out

#plot of pca variances
screeplot(pca.train,type="line",main="Scree Plot")

#contributions of variables to PC1:
fviz_contrib(pca.train, choice = "var", axes = 1, top = 50)

#contributions of variables to PC2:
fviz_contrib(pca.train, choice = "var", axes = 2, top = 20)

#contributions of variables to PC3:
fviz_contrib(pca.train, choice = "var", axes = 3, top = 50)

#contributions of variables to PC4:
fviz_contrib(pca.train, choice = "var", axes = 4, top = 50)

#contributions of variables to PC5:
fviz_contrib(pca.train, choice = "var", axes = 5, top = 50)

#now checking PC's 1 to 4 because they explain the most variance
fviz_contrib(pca.train, choice = "var", axes = 1:4, top = 50)
#we see that "hair", "overall, "skin" are the most important and then 
#the graph tapers off to other variables

#another visualization where "skin", "overall" and "hair" standout
fviz_mca_var(pca.train, col.var = "contrib",
            gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             #too avoid text overlapping (is slow)
            ggtheme = theme_minimal()
)

#extract the results for individuals
indiv <- get_pca_ind(pca.train)
indiv

#coordinates of individuals
head(indiv$coord)

#these are the active variables with the highest cos2
fviz_pca_var(pca.train, repel = TRUE, radient.cols = c("white", "blue", "red"), select.var= list(cos2 = 30))
#the variables listed are what we will reduce our variables down to:
#"overall"
#"moistur"
#"sensit"
#"dri"
#"cur"
#"blow"
#"straight"
#"hair"
#"curl"
#"style"
#"dryer"
#"heat"
#"iron"
#"flat"
#"item"
#"order"
#"seller"
#"return"
#"receiv"
#"amazon"
#"compani"
#"money"
#"wrinkle"
#"notic"
#"use"
#"face"
#"week"
#"cream"
#"skin"
#"acn"
#"thick"


#REDUCING FEATURES 
#using only features from above list of variables - training dataframe
newfeat1 <- combineddf.train[, (names(combineddf.train) %in% c("helpful","overall", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
                                                               "return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn", "thick"))]

#will also divide the combineddf.test dataframe in the same way - test datraframe
newfeat1.test <- combineddf.test[, (names(combineddf.train) %in% c("helpful","overall", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
                                                                                "return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn", "thick"))]


#WE WILL ALSO DO ANOTHER SEGMENTATION OF JUST THE TFIF SCORES OF TERMS WE CHOSE

#using only word features from above list of terms - training datafram 
words.train <- combineddf.train[, (names(combineddf.train) %in% c("helpful", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
                                                               "return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn", "thick" ))]

#will also divide the combineddf.test dataframe in the same way - test dataframe
words.test <- combineddf.test[, (names(combineddf.train) %in% c("helpful", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
                                                                   "return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn", "thick"))]


#subsetting columns 
newfeat1 <- combineddf.train[, (names(combineddf.train) %in% c("helpful","overall", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
                                                               "return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn"))]


#NORMALIZE DATA

#creating normalize function
normalform <- function(x) {(x - min(x, na.rm=TRUE))/(max(x,na.rm=TRUE) - min(x, na.rm=TRUE))}

#using lapply to apply normalform() to every column in train data
normed.train <- as.data.frame(lapply(newfeat1, normalform))
#check that every that the values are within 0 to 1
lapply(normed.train, range)


#using lapply to apply normalform() to every column in a test data
#first taking only relevant variables
nowt.test <- combineddf.test[c(1:2,4:551)] 
normed.test <- as.data.frame(lapply(nowt.test, normalform))
#check that every value is within 0 to 1
lapply(normed.test, range)

#LOGISTIC REGRESSION

justwords.glm <- glm(helpful~. ,family=binomial(link='logit'), data=normed, control = list(maxit = 100))

#evaluate logistic regression model
predict_glm <- as.numeric(predict(justwords.glm, normed.test, type="response") > 0.5)
table(normed.test$helpful,predict_glm,dnn=c("Observed","Predicted"))

#finding accuracy of model
classiferror <- mean(predict_glm != normed.test$helpful)
accu <- paste('Accuracy',1-classiferror)
accu
#Accuracy 0.8815612382234192 ie. 88.2%

#A DOWNSAMPLED LOGISTIC REGRESSION

#defining "not in" func
'%ni%' <- Negate('%in%')  

#prevents printing sci notations.
options(scipen=999)  

set.seed(100)
down.train.lg <- downSample(x = normed.trainfactor[, colnames(normed.trainfactor) %ni% "helpful"],
                         y = normed.trainfactor$helpful)

#there are now 5801 reviews for each class
table(down.train.lg$Class)

#fitting model
downsampled.glm <- glm(Class~. ,family=binomial(link='logit'), data=down.train.lg, control = list(maxit = 100))

#summary of model
summary(downsampled.glm)

#evaluate logistic regression model
predict_down_glm <- as.numeric(predict(downsampled.glm, normed.test, type="response") > 0.5)
table(normed.test$helpful,predict_down_glm,dnn=c("Observed","Predicted"))


#        Predicted
#Observed    0    1
#       0 1014  364
#       1 2730 7780

#finding accuracy of new model
down_lg_classiferror <- mean(predict_down_glm != normed.test$helpful)
down_accu <- paste('Accuracy',1-down_lg_classiferror)
down_accu 
#"Accuracy 0.739737550471063" - ie. 73.9% 


#NAIVE BAYES

#convert helpful to as.factor as helpful was numeric in train dataset
nb <- naiveBayes(as.factor(helpful)~., data=normed.train)
nb_pred <- predict(nb, normed.test, type="class")
table(nb_pred, normed.test$helpful, dnn=c("Prediction","Actual"))

#         Predicted
#Observed    0    1
#0          496 1743
#1          929 8720

#finding accuracy of new model
nb.classiferror <- mean(nb_pred != normed.test$helpful)
nb.accu <- paste('Accuracy',1-nb.classiferror)
nb.accu
#Accuracy = 0.775235531628533 ie. 77.5%


#RANDOM FOREST

#random Forest model using downsampling technique of helpful class
ctrl <- trainControl(method = "repeatedcv", 
                     number = 2, 
                     repeats = 2, 
                     verboseIter = FALSE,
                     sampling = "down")


model_rf_under <- caret::train(helpful ~ .,
                                data = normed.trainfactor,
                                method = "rf",
                                trControl = ctrl)

#evaluating downsampled random forest model
testforest = predict(model_rf_under, newdata=normed.testfactor)
#confusion matrix downsampled model
table(testforest, normed.testfactor$helpful, dnn=c("Observed","Predicted"))
#testforest     
#           Predicted
#Observed    0    1
#        0 1079 2978
#        1  346 7485
#when downsampled in random forest we were not better able to predict reviews that are not helpful - see below model.
#true negative rate - 0.2659601 - ie. 26.6%
#TN/N = 1079/(1079 + 2978)

rf_accu <- (7485 + 1079)/(7485 + 2978 + 346 + 1079)
rf_accu
# Accuracy 0.757193 ie. 75.7%

#random forest model without downsampling
#note: randomForest() will default to classification or regression depending on the variable.
#we change helpful to as factor to accommodate

rf1 <- randomForest(helpful~., data = normed.trainfactor,  ntree = 100, mtry = 5, importance = TRUE)
#lowered for computation reasons

testforest_notds = predict(rf1, newdata=normed.testfactor)
#confusion matrix downsampled model
table(testforest_notds, normed.testfactor$helpful, dnn=c("Observed","Predicted"))

#            Predicted
#Observed     0     1
#0           162   196
#1           1263 10267

#true neg rate - tn/n 
#162/(162 + 196) = 45.2% 
#true neg rate is better than when we downsampled
