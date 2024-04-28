Using “Amazon Product Data” for the “Beauty” category collected by Julian McAuley[1], we will endeavor to learn the characteristics of helpful customer reviews and see if we can predict the helpfulness of reviews.


EXPLORATORY ANALYSIS & DATA PREPARATION

ATTRIBUTE CHARACTERISTICS AND DESCRPTIVE STATISTICS

There are 2,023,070 Beauty product reviews included in the original dataset.  The original dataset was formatted in loose json and had to be converted to strict json using a python script which can be found on the project github as (see Appendix). After this was done, the data was then streamed into R for processing.
The 9 variables included in the original dataset:
•	reviewerID – the ID of the reviewer
•	asin – the ID of the product
•	reviewerName – the name of the reviewer
•	helpful – the helpfulness rating of the review – i.e. 4/5. the numerator is the number of ratings that say the review is helpful and the denominator is the total amount of people who have rated the review
•	reviewText – the text of the review
•	overall – the rating in stars that the reviewer gives the product
•	summary – summary of the review
•	unixReviewTime – time of the review (in unix time)
•	reviewTime – time of the review (raw)
sample entry:
 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/fd825695-5f8c-4e30-98b0-dd99f44a46ea)


The problem we are solving for (ie. what we want to predict) is whether a review is either “helpful” or not “helpful. The “helpful score” represents the following: Amazon users can rate a review as either “helpful” or “not helpful” and our dataset records “helpful” as an array. To train our model, we will also create a new feature for “helpful” using a threshold that we set - more on this the next section.

FEATURE SELECTION
   
The features we will keep from the original dataset are as follows:
•	reviewText – this will be used to generate features using natural language processing.
•	helpful  – we will be using this column to create two new features as well as the class label. The new “helpful” class lab that we create (ie. labels of helpful or not helpful) will be used to train our model. 
•	 Overall – the overall star rating of the product by the reviewer.
We have excluded the other features as we will focusing on NLP of the review text in our approach. 
The first new feature we create is “helpful_num” which isolates the numerator from the “helpful” column. This feature will represent the amount of votes for the given review being “helpful”. We also create a second feature called “helpful_denom” which isolates the denominator from the “helpful” column. This feature will represent the total amount of votes on a product.
We check for missing data and there aren’t any. We also clean up the unneeded characters in the columns (ie. brackets and commas). We also need to change the helpful_denom and helpful_num features to “numeric” so that we can gather correlations and do operations on the columns. The code for the full cleaning process can be found on the project github (see Appendix).
We then find the dataset’s mean, standard deviation, median, min and max. See below graphic.

 
![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/505bad16-8dc3-499c-8eb3-e220b0269ce2)


 
Of particular concern is the max values in “helpful_num” and “helpful_denom” which is very high above the mean in their respective standard deviations. We will take a closer look at this see how we should deal with any outliers as we begin to train the model – for now we keep things as is.
 
![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/bd3c05d1-c159-4e6d-8e1b-68af848f2e9e)


Because we don’t want to artificially class “helpful” reviews as non-helpful just because they are not rated by enough people, we decide to only use observations which have a helpful_denom column that is more than 10. Meaning that, more than 10 people had to rate the review for us to consider that rating valid and to include it in our dataset. This brings are amount of observations down to 59,43f9.
We then create the new class feature for “helpful”. Reviews that were considered  “helpful” are given a value of 1 and those “not helpful” are given 0. We decided that the cut off for a review to be considered helpful was that at least 60% of raters had to rate the review as “helpful” (ie. a ratio of 0.6). We find this ratio for each observation by dividing the “helpful_num” column by the “helpful_denom” column. We will use this feature as the class label and train our model to predict for this label.
We also check to see the balance of the helpful vs. not helpful reviews and we find that  our dataset has significantly more helpful reviews. When we are training our model, we will need to be mindful of this and think of different sampling methods to deal with the issue as both categories need representation to build a useful model. 

 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/d98bad53-fb1e-4a2a-b085-df669a7434b6)
 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/711ed59e-f42f-4540-b76a-074117361e65)



 
We also check distributions of the “overall” feature – which is the overall star score given to products by reviewers. This may be meaningful later when training our model as it looks like this category also has a slight correlation to the “helpful” class label.
We find that there is a small correlation between the 'helpful' class label and amount of helpful ratings (ie. helpful_num) as we plot correlation of the dataset.

DISTRIBUTION OF QUANTITATIVE ATTRIBUTES

To check for normality, we use the Anderson-Darling normality test, which works for larger sample sizes where the Shapiro does in R. The Anderson-Darling test rejects the hypothesis of normality when the p-value is less than or equal to 0.05. If we fail the normality test we can - with 95% confidence – say the the data does not fit the normal distribution. 
The p-value found for “helpful_denom” is  < 2.2e-16  and due to this we reject the null hypothesis and  find no significant departure from normality. The same p-value was found for “helpful_denom” and for “overall”. 
We also examine the correlation between variables and find that the “overall” feature (the number of stars the reviewer has given the item) is lightly correlated to the class label of “helpful”. 

 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/b7fb823f-3a61-47fb-a25e-ddc398728e1c)





TEXT MINING

To be able to take a closer look at the review text we used text mining tools in R. Using the TM package, we transformed the review text. 
 
This portion was divided into the following sub-steps:
a)	All text was changed into lower case
b)	Numbers were removed as well as punctuation. 
c)	Stop words (a built in to the definition – words like “the” and “and”) are removed.
d)	White space is removed and the words are then stemmed meaning that suffixes a portion of the word was cut off – for example, running and runner would be stemmed to “run” as it extracts the sentiment and reduces necessary sparsity in the corpus. 
e)	All of this is then put into a corpus for us to further transform.

We then created a document-term matrix where documents are rows and the words are columns. The frequency of the term in the document are entries. Due to the sparsity being very high, we removed low frequency words as they were possibly typos or otherwise uninteresting for our purposes.

After the above pre-processing, we then extracted the following values: term frequency, inverse document frequency (how important the term is),  idf(t) (total number of documents relative to the amount of documents with the term within it), tf(t) (number of times a term appears relative to total number of terms).

The higher the tf-idf score, the more rare the word is and the same is true to other way around. A small score means the term is more frequent. We created a new dataframe with each term and their tf-idf scores. This process creates 545 new variables – one for each of the words. This was then binded this to the data set. 

Find a word cloud of the words in the tf-idf matrix below:

 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/e1c74ead-9a57-40fe-b127-8df971e32ce9)




FURTHER FEATURE SELECTION USING PRINCIPAL COMPONENT ANALISYS


We only applied PCA to the training dataset as we didn’t want to influence the testing dataset. The training dataset was 80% of the full dataset and we tested models on the remaining 20%. The transformed dataset was now 551 column data set, and we wanted to reduce the features significantly for the sake of computation and comprehension.

The full code can be found on the github link in the project Appendix, however the main point is that we used the first four principal components for analysis as these explained the most amount of variance while having a manageable amount of features. With a very wide data set computation time was a big concern. These components combined explain the most amount variance in the data set. We chose to use the first four principal components.
 

 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/6ebb8eaa-3df2-4eff-8969-9b4a3003079b)

 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/56831703-22a3-4367-bce4-a502902321c5)

![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/a78ffd1a-4b3a-4c80-bdaf-fa86a62603a1)








 




The variables that will be kept in our training data set are as follows:  "overall", "moistur", "sensit", "dri","cur","blow","straight", "hair","curl","style","dryer","heat","iron","flat","item","order","seller",
"return", "receiv", "amazon", "compani", "money", "wrinkle", "notic", "use", "face", "week", "cream", "skin", "acn", “thick”.


NORMALIZATION

We normalize both training and test data sets before moving forward. Data normalization is the process of converting the range of values for a continuous attribute to 0 to 1.  This converts the largest value in the attribute’s range to 1 and the smallest value to 0. This is a technique that is often required by various algorithms and when you know that the distribution is not normal.


MODELS

LOGISTIC REGRESSION
Logistic Regression is a classification algorithm that is used to predict a binary outcome. In order to use Logistic Regression, the dependent variable should have mutually exclusive and exhaustive categories. This applies to our dataset as the dependent variable is binary: helpful or not - 1 or 0.  

Call:
glm(formula = helpful ~ ., family = binomial(link = "logit"), 
    data = normed.train, control = list(maxit = 100))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-3.3484   0.2764   0.3070   0.4302   1.8823  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  0.73977    0.02911  25.409  < 2e-16 ***
overall      2.36162    0.03808  62.020  < 2e-16 ***
skin        -0.03844    0.22428  -0.171 0.863917    
order       -1.99485    0.28495  -7.001 2.55e-12 ***
cream        0.21604    0.32642   0.662 0.508058    
face         0.42442    0.26651   1.593 0.111269    
moistur      0.79012    0.32730   2.414 0.015777 *  
notic       -0.34002    0.42819  -0.794 0.427143    
use         -0.89458    0.18473  -4.843 1.28e-06 ***
week         0.08350    0.27805   0.300 0.763944    
amazon       0.60154    0.24001   2.506 0.012199 *  
dri          1.89435    0.43074   4.398 1.09e-05 ***
item        -2.19723    0.40202  -5.465 4.62e-08 ***
sensit       1.45672    0.55978   2.602 0.009260 ** 
hair        -0.39342    0.20285  -1.939 0.052448 .  
seller       1.82403    0.42623   4.279 1.87e-05 ***
money       -0.72958    0.25041  -2.914 0.003574 ** 
receiv      -1.32221    0.36135  -3.659 0.000253 ***
compani      0.67368    0.31126   2.164 0.030439 *  
return       0.55472    0.30280   1.832 0.066953 .  
style        4.39649    1.09128   4.029 5.61e-05 ***
flat        -0.10744    0.87347  -0.123 0.902104    
heat         2.94539    0.69520   4.237 2.27e-05 ***
iron         1.57433    0.70180   2.243 0.024880 *  
straight     1.93599    0.95669   2.024 0.043008 *  
cur          1.28948    0.92206   1.398 0.161970    
thick        1.97905    0.55292   3.579 0.000345 ***
dryer        1.79969    0.58440   3.080 0.002073 ** 
blow         1.13643    0.78051   1.456 0.145388    
curl         1.46862    0.60163   2.441 0.014643 *  
acn         -0.20174    0.49192  -0.410 0.681732    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 35272  on 47550  degrees of freedom
Residual deviance: 29650  on 47520  degrees of freedom
AIC: 29712

Number of Fisher Scoring iterations: 6
Number of Fisher Scoring iterations: 6

In our logistic regression model we find that the most significant variables in relation to helpfulness are “overall”, “order”, “use”,  “dri”, “item”, “item”, “seller”, “receive”, “heat” and “thick. 

Confusion Matrix
![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/907894cb-546a-43ed-a6e7-371619f7c3d3)

 
Accuracy: 88%

True Positive Rate (ie. sensitivity or recall) :
TP/P = 10429/(34 + 10429 ) =  99.7%

True Negative Rate (ie. specificity): 
TN/N  = 51/(51 + 1374) = 3.5%

Precision (ie. positive predictive value?):
TP/TP + FP
10429/(10429 + 1374) = 88.4%

We then tried to downsample the dominant class of helpful reviews and include the same amount of helpful and unhelpful reviews. 

Call:
glm(formula = Class ~ ., family = binomial(link = "logit"), data = down.train.lg, 
    control = list(maxit = 100))

Deviance Residuals: 
     Min        1Q    Median        3Q       Max  
-2.54882  -0.76535   0.04914   0.77878   2.32380  

Coefficients:
            Estimate Std. Error z value             Pr(>|z|)    
(Intercept) -1.21452    0.04459 -27.238 < 0.0000000000000002 ***
overall      2.38868    0.05254  45.468 < 0.0000000000000002 ***
skin         0.30948    0.29204   1.060             0.289270    
order       -1.76871    0.44959  -3.934            0.0000835 ***
cream        0.26813    0.44556   0.602             0.547315    
face         0.37331    0.35192   1.061             0.288786    
moistur      1.12429    0.39175   2.870             0.004106 ** 
notic       -0.28545    0.58826  -0.485             0.627498    
use         -1.00320    0.25548  -3.927            0.0000861 ***
week         0.08076    0.37616   0.215             0.830013    
amazon       0.45433    0.36055   1.260             0.207624    
dri          2.22460    0.51830   4.292            0.0000177 ***
item        -2.56445    0.70367  -3.644             0.000268 ***
sensit       0.87132    0.67554   1.290             0.197115    
hair        -0.46722    0.27965  -1.671             0.094781 .  
seller       0.67028    0.61786   1.085             0.277992    
money       -1.14626    0.43972  -2.607             0.009139 ** 
receiv      -2.25762    0.61783  -3.654             0.000258 ***
compani     -0.07364    0.48868  -0.151             0.880218    
return       0.35127    0.48153   0.729             0.465705    
style        3.85219    1.24982   3.082             0.002055 ** 
flat        -1.04603    1.19305  -0.877             0.380612    
heat         3.45306    0.83748   4.123            0.0000374 ***
iron         2.14760    0.89302   2.405             0.016178 *  
straight     0.51959    1.05804   0.491             0.623369    
cur          1.61536    0.99195   1.628             0.103427    
dryer        0.92256    0.71992   1.281             0.200027    
blow         1.46784    0.94186   1.558             0.119128    
curl         1.30269    0.72089   1.807             0.070753 .  
acn          0.13940    0.64112   0.217             0.827878    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 16084  on 11601  degrees of freedom
Residual deviance: 12956  on 11572  degrees of freedom
AIC: 13016

Number of Fisher Scoring iterations: 4

The new model had a lower accuracy, which makes sense because we’re taking away the information from thousands of helpful reviews that the model could have used to learn. However, the true negative rate increased to 73.% when we downsampled to make both the helpful and helpful reviews even (ie. 5801 reviews for each class). In the model using the downsampled dataset, the most significant variables in relation to helpfulness were the variables are “overall”, “order”, “use”, “dri”, “item”, “receiv”, and “heat”.

Confusion Matrix


 ![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/167cb759-5420-4274-8c53-311d5eb6f346)



Accuracy: 73.9%

True Positive Rate (ie. sensitivity or recall) :
TP/P =  7780/(2730 + 7780 ) = 74%

True Negative Rate (ie. specificity): 
TN/N =  1014/(1014 + 364) = 73.5% 

Precision (ie. positive predictive value?):
TP/TP + FP
10429/(10429 + 1374) = 88.4%



NAIVE BAYES

Naive Bayes is a supervised machine learning classification method based on Bayes’ Theorem. The assumption for Naive Bayes is that all attributes are equally important and are statistically independent from each other. These assumptions may not necessarily be true, but in practice the algorithm often provides reasonably good results. 

Confusion Matrix

![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/28e3b237-03fe-4f60-adbe-53d8b5cea0c8)

 

Accuracy: 77.5%

True Positive Rate (ie. sensitivity or recall) :
TP/P = 8720/(929 + 8720) = 90.4%

True Negative Rate (ie. specificity): 
TN/N = 496/(496 + 1743) = 22%

Precision (ie. positive predictive value?):
TP/TP + FP
8720/(8720 + 1743) = 83.3%


RANDOM FOREST
For our implementation of random forest we decided to downsample helpful reviews in hopes of increasing our true negative rate which has been poor in our previous two models. We used two fold validation in our model, as well as two sampling interations. These were chosen for lowered computation times.

Confusion Matrix

![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/7c109bc0-9f55-4187-87ac-5e7af72fd91d)

 

Accuracy: 75.7%
True Positive Rate (ie. sensitivity or recall) :
TP/P = 7485/(346 + 7585) = 94.3%

True Negative Rate (ie. specificity): 
TN/N = 1079/(1079 + 2978) = 26.6%
 
Precision (ie. positive predictive value?):
TP/TP + FP
8720/(8720 + 1743) = 83.3%


CONCLUSIONS AND RECOMMENDATION

Strengths and Weaknesses of Each Model


![image](https://github.com/114141/predicting-helpfulness-ratings-of-amazon-product-reviews/assets/12577888/7106497f-1d5b-4309-b1dd-31f2e9974e03)



Recommendations and Final Thoughts

The research question for this project was whether we can learn what makes a review more likely to be voted as helpful, however as we progressed through the project and saw that our models had a harder time predicting unhelpful reviews the original question became harder to find answers to. 

The true positive rate for all models was high, with logistic regression on an unbalanced dataset being the highest at 99.7%. However, this was mostly at the cost of mis-categorizing unhelpful reviews. Even when we balanced the dataset by downsampling the dominant class (helpful) and logistic regression was able to raise its true negative rate to 73.5%, we fear that this came at the cost of truly learning what makes a review unhelpful. Information was lost when we downsampled. More work needs to be done here to either gather more data on unhelpful reviews or look into creating other features. We are generally unhappy with the specificity rate across the board. 

If we had more time, we would try to do more feature engineering and change our methods of data preparation. Perhaps finding other ways of looking at the variables in the original data set that we immediately decided to cut out. 

We are also curious as to how different our models would have been if the dataset was more recent, or if the merchant (Amazon) asked the customer different kinds of questions to determine if the review was helpful. However, ultimately, we do think our models would have been more accurate if they contained more cases of unhelpful reviews. A different way of approaching this problem would be to focus on which customers have had their buying decision influenced by a customer review. 

If we had to recommend one algorithm to use in measuring the helpfulness of reviews however, it would be logistic regression because of the accuracy, true positive and true negative rates.











