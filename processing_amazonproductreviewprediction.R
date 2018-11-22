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
#library(stringr)
#library(tidyr)
#library(rjson)
#library(jsonlite)
#library(data.table)
#library(psych)
#library(ggplot2)
#library(corrlot)
#library(plyr)
#library(tm)
#library(tidytext)
#library(tidyverse)
#library(SnowballC)
#library(dplyr)

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

#FEATURE SELECTION AND CREATING NEW FEATURES

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
missing <- newdf[!complete.cases(newdf),]
#there are no mising values
missing

#check classes again
sapply(newdf, class)

#since helpful_num and helpful_denom are char change them to numeric
newdf$helpful_num <- as.numeric(as.character(newdf$helpful_num))
newdf$helpful_denom <- as.numeric(as.character(newdf$helpful_denom))

#describe the dataset
describe(newdf)
#note that maximum numbers are very high above mean in standard deviations which indicates that there are outliers which could throw off model

#EXPLORATORY DATA ANALYIS AND CREATING A CLASS LABEL

#boxplot of distrubution of data
names = c("overall","helpful_num","helpful_denom")
newplot <- boxplot(newdf$overall, newdf$helpful_num, newdf$helpful_denom, names=names)
newplot

#we will only use records where the amount of raters (ie. denominator column) is more than 10 so that
#we don't artifically rate "helpful" reviews as non-helpful just because they are not rated by people.
#saving this to new dataframe which we will use going forward
ten_newdf <- newdf[which(newdf[,2]>10),]
ten_newdf

#now creating a new feature - ie. the class label for "helpful".
ishelpful = 0.6
ten_newdf$helpful <- ifelse((ten_newdf$helpful_num/ten_newdf$helpful_denom)>ishelpful, 1, 0)

#checking to see how imbalanced the dataset is. it's really impalanced towards "helpful" reviews
counthelpful <- count(ten_newdf, "helpful")
counthelpful

#using corrplot() to visualize correlation
subsetforcor <- ten_newdf[c(1,2,4,5)]
corrplot(cor(subsetforcor), method="color")
#There is a small correlation between the 'helpful' rating and our score.

#plotting histograms
overall <- ten_newdf$overall
hist(overall)

#cleaning text as a part of preprocess (using tm package)
reviews <- ten_newdf$reviewText

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
review_doctm <- removeSparseTerms(review_doctm, 0.98)
review_doctm

#inspecting five rows of matrix again
inspect(review_doctm[305:310, 305:310])

#finding tf-idf which finds the relative importance of a word to a document.
reviewdoctm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
#now creating tf-idf scores
reviewdoctm_tfidf = removeSparseTerms(reviewdoctm_tfidf, 0.98)
#note have increased threshhold to bring in more terms
reviewdoctm_tfidf

#checking the first document
inspect(reviewdoctm_tfidf[1,1:15])

#converting to dataframe
tfidfam <- as.matrix(reviewdoctm_tfidf)
tfidfdf <- as.data.frame(tfidfam)
#checking out how many terms are in the dataframe
ncol(tfidfdf)
#there are 545 terms

#creating new dataframe for reviews and tf-idf to inspect before adding as feature to full dataset
n_reviews <- cbind(reviews, tfidf)
#view sample of new dataframe
head(n_reviews)

#binding tfidfdf with working dataframe
combineddf <- cbind(ten_newdf, tfidfdf)

#checking class of new column
sapply(combineddf, class)

#finding the top words - takes lower frequency bound as argument
toptfidf <- findFreqTerms(reviewdoctm_tfidf, 100)
toptfidf

#LOGISTIC REGRESSION MODEL

#subset just terms, helpful and wordcount columns
justwords <- combineddf[c(5:551)] 

#spliting train and testing sets for just words and helpful df
trainrows <- sample(nrow(justwords),nrow(justwords)*0.80)
justwords.train = justwords[trainrows,]
justwords.test = justwords[-trainrows,]

#building model
justwords.glm <- glm(helpful~. ,family=binomial(link='logit'), data=justwords.train, control = list(maxit = 100))

#evaluate logistic regression model using confusion matrix
predict_glm <- as.numeric(predict(justwords.glm, justwords.test, type="response") > 0.5)
table(justwords.test$helpful,predict_glm,dnn=c("Observed","Predicted"))

#finding accuracy of new model
classiferror <- mean(predict_glm != justwords.test$helpful)
accu <- paste('Accuracy',1-classiferror)
accu
#"Accuracy 0.883748317631225"














