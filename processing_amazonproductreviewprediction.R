#install.packages("rjson") 
#install.packages("jsonlite")
#insall.packages("tidyr")
#install.packages("data.table")
#install.packages("stringr")
#install.packages("psych")
#install.packages("ggplot2")
#install.packages("corrplot")
#install.packages("plyr")
#library(stringr)
#library(tidyr)
#library(rjson)
#library(jsonlite)
#library(data.table)
#library(psych)
#library(ggplot2)
#library(corrlot)
#library(plyr)

#note: original dataset is not in strict json see python code to turn the data set into strict json

#uploading json new_reviews into data frame 
json_reviews_amazon <- jsonlite::stream_in(file("/Users/annagebre/new_reviews_beauty.json"))

#check datatypes of attributes of eview data
sapply(json_reviews_amazon, class)

#checking to examine first like (note delete "3" to show 6 rows instead))
head(json_reviews_amazon, 3)

#subsetting the variables we are going to use in our model - "helpful", "reviewtext" and "overall"
newdf <- json_reviews_amazon[c(4, 6:7)]

#checking that we've subsetted correctly
head(newdf, 3)

#creating new features - helpful numerator and helpful denominator

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

#maximum numbers are very high above mean in standard deviations which indicates that there are
#outliers which could throw off model

#EXPLORATORY DATA ANALYIS

#boxplot of distrubution of data
names = c("overall","helpful_num","helpful_denom")
newplot <- boxplot(newdf$overall, newdf$helpful_num, newdf$helpful_denom, names=names)
newplot

#decided to only use records where the amount of raters (ie. denominator column) is more than 10 so that
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
#There is small correlation between the 'helpful' rating and our score.

















