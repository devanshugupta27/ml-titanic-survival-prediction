---
title: "Titanic Survival Prediction"
author: "Devanshu Gupta"
date: "August 25, 2017"
output:
  pdf_document: default
  
---

# 1 Introduction
##This is my first stab at a Kaggle script. I will focus on doing some illustrative data visualizations along the way. I'll then use randomForest in a different way to create a model predicting survival on the Titanic. I am new to machine learning in R and hoping to learn a lot, so feedback is very welcome!
There are 3 parts as:

  * Data Munging and Cleaning
  * EDA
  * Prediction


## 1.1 Loading and examining the dataset and its variables.

```r
#Loading the necessary libraries
library(ggplot2)
library(lattice)
library(caret)
library(ranger)
library(dplyr)
library(e1071)
```


```r
#Reading the train and the test datasets.
train <- read.csv("train.csv",stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
```

```r
#Checking the structure of the dataset
str(train)
```

```
## 'data.frame':	891 obs. of  12 variables:
##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
##  $ Sex        : chr  "male" "female" "female" "female" ...
##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
##  $ Cabin      : chr  "" "C85" "" "C123" ...
##  $ Embarked   : chr  "S" "C" "S" "S" ...
```

```r
#Examining the summary of the dataset
summary(train)
```

```
##   PassengerId       Survived          Pclass          Name          
##  Min.   :  1.0   Min.   :0.0000   Min.   :1.000   Length:891        
##  1st Qu.:223.5   1st Qu.:0.0000   1st Qu.:2.000   Class :character  
##  Median :446.0   Median :0.0000   Median :3.000   Mode  :character  
##  Mean   :446.0   Mean   :0.3838   Mean   :2.309                     
##  3rd Qu.:668.5   3rd Qu.:1.0000   3rd Qu.:3.000                     
##  Max.   :891.0   Max.   :1.0000   Max.   :3.000                     
##                                                                     
##      Sex                 Age            SibSp           Parch       
##  Length:891         Min.   : 0.42   Min.   :0.000   Min.   :0.0000  
##  Class :character   1st Qu.:20.12   1st Qu.:0.000   1st Qu.:0.0000  
##  Mode  :character   Median :28.00   Median :0.000   Median :0.0000  
##                     Mean   :29.70   Mean   :0.523   Mean   :0.3816  
##                     3rd Qu.:38.00   3rd Qu.:1.000   3rd Qu.:0.0000  
##                     Max.   :80.00   Max.   :8.000   Max.   :6.0000  
##                     NA's   :177                                     
##     Ticket               Fare           Cabin             Embarked        
##  Length:891         Min.   :  0.00   Length:891         Length:891        
##  Class :character   1st Qu.:  7.91   Class :character   Class :character  
##  Mode  :character   Median : 14.45   Mode  :character   Mode  :character  
##                     Mean   : 32.20                                        
##                     3rd Qu.: 31.00                                        
##                     Max.   :512.33                                        
## 
```

The variables of the following dataset are as:

1. PassengerId - Id of each passenger.
2. Survived    - Describing whether passenger survived or not.
                      0 - Not Survived, 1 - Survived
3. Pclass      - Class of each passenger 
4. Name        - Name of the passenger
5. Sex         - Sex of the passenger
6. Age         - Age of the passenger
7. SibSp       - Number of siblings/spouses aboard
8. Parch	     - Number of parents/children aboard
9. Ticket      - Ticket number
10. Fare	     - Fare
11. Cabin	     - Cabin
12. Embarked	 - Port of embarkation

#2 Data Munging and Cleaning
## 2.1 Changing type to factor

Various variable in the dataset should be represented as factors but are represented as numeric, which does not make any sense.

These variables are:
* Survived
* Pclass
* Sex
* SibSp
* Parch
* Embarked


```r
#Converting Survived to a factor 
train$Survived <- factor(train$Survived)

#Converting Pclass to a factor
train$Pclass <- factor(train$Pclass)

#Converting Sex to a factor
train$Sex  <- factor(train$Sex)

#Converting SibSp to a factor
train$SibSp <- factor(train$SibSp)

#Converting Parch to a factor
train$Parch <- factor(train$Parch)

#Converting Embarked to a factor
train$Embarked <- factor(train$Embarked, ordered = FALSE)
```
## 2.2 Name Issues

 The Name of the passengers has a issue. The naming convention also seems to be somewhat archaic and scores low on human readability. 

For females, their original name is the one inside the "()". 
For Example: The original name of **Cumings, Mrs. John Bradley (Florence Briggs Thayer)** is **"Florence Briggs Thayer"**.
For males, their original name is as:
Original name of the person named in dataset as **"Braund, Mr. Owen Harris"** is **"Owen Harris Braud"**

 The name can be converted by means of a function involving a if-else statement check.


```r
head(train$Name)
```

```
## [1] "Braund, Mr. Owen Harris"                            
## [2] "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"
## [3] "Heikkinen, Miss. Laina"                             
## [4] "Futrelle, Mrs. Jacques Heath (Lily May Peel)"       
## [5] "Allen, Mr. William Henry"                           
## [6] "Moran, Mr. James"
```
We can see that names are not as we use names now.

```r
convert_name <- function(name) {
  
  if (grepl("\\(.*\\)", name)) {           # women: take name from inside parentheses
    gsub("^.*\\((.*)\\)$", "\\1", name)
  } else {                                # men: take name before comma and after title
    gsub("^(.*),\\s[a-zA-Z\\.]*\\s(.*)$", "\\2 \\1", name)
  }
}
#grepl() searches for pattern and is gives a logical result. 
#gsub(pattern, replacement, string) is used to replace every occurence of pattern in the string with the replacement.  

###The pattern is as :
#  * ^        denotes starting of pattern
#  * .*       denotes occurence of any character zero or more times.
#  * \\(      denotes that we are actually looking for '(' in the string. Names of females of the dataset are inside paranthesis.
#  * (.*)     denotes a back-reference.
#  * \\)      denotes we actually want to look for ) in the string.
#  * \\1      denotes first back-reference. For every occurence of (.*), there is a back-reference 
#  * \\s      matches a space
# * [a-zA-z] This sprecifies character ranges. All characters in a-z and A-Z are matched.
#  * $        denotes end of string.
```
Calling the function convert_name to the passenger name.

```r
pass_names <- train$Name

clean_pass_names <- vapply(pass_names, FUN = convert_name,
                           FUN.VALUE = character(1), USE.NAMES = FALSE)

train$Name <-  clean_pass_names

#The function is applied to pass_names (The vector that contains all the names of the train dataset) via vapply so as to use convert_names for all names in the dataset.

head(train$Name)
```

```
## [1] "Owen Harris Braund"     "Florence Briggs Thayer"
## [3] "Laina Heikkinen"        "Lily May Peel"         
## [5] "William Henry Allen"    "James Moran"
```
We can see that the names of the passengers have been cleaned.

# 3 Exploratory Data Analysis
## 3.1 Pclass v/s Survived


```r
train %>% 
    ggplot(aes(x = Pclass, fill = Survived)) + 
          geom_bar()
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png)

### Inference from the Graph 
From the graph, it is clear that the number of passenger who survived is independent on the Class of passenger, while the number of passenger who couldn't survived seems to be dependent on the class of passenger. 
Same can be established by the the table function.  

```r
tab <- table(train$Pclass, train$Survived)
prop.table(tab,1)
```

```
##    
##             0         1
##   1 0.3703704 0.6296296
##   2 0.5271739 0.4728261
##   3 0.7576375 0.2423625
```
So while 62.96% of passenger from Class 1 survived, only 24.23% passenger belonging to the Class 3 could survive.

##3.2 Sex v/s Survived


```r
train %>%
      ggplot(aes(x = Sex, fill = Survived)) + 
            geom_bar(stat = "count", position = "fill")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png)

### **Inference from the Graph** 
From the graph, it is clear that a large propotion of female passengers survived while only a thin population of male passengers could survive, eventhough the number of male passenger aboard was almost two times of the female passengers.

Same can be established by the the table function.  

```r
tab <- table(train$Sex, train$Survived)
prop.table(tab,1)
```

```
##         
##                  0         1
##   female 0.2579618 0.7420382
##   male   0.8110919 0.1889081
```
So while 74.20% of female passenger survived, only 18.89% of the male passenger could survive.

##3.3 Age v/s Survived

```r
train %>%
      ggplot(aes(x = Age, fill = Survived)) + 
            geom_histogram()
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

```
## Warning: Removed 177 rows containing non-finite values (stat_bin).
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

### Inference from the Graph 

While there seems an uniform trend among age on the number of survivals, some inferences can be deduced.
The rate of survival among infants was high.
Also, most of the passenger belong to the 20-40 year age group.

##3.4 Embarked v/s Survived

```r
train %>%
  filter(Embarked %in% c("S","C","Q")) %>%
  ggplot() +
  geom_bar(aes(Embarked, fill = Pclass), position = "dodge") +
  facet_grid(~ Survived)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

## Inference from Graph
Proportion of passengers survived seems to be equal for passenger from port Q and S. 
The same can be established from the prop table.

```r
tab <- table(train$Embarked, train$Survived)
prop.table(tab,1)
```

```
##    
##             0         1
##     0.0000000 1.0000000
##   C 0.4464286 0.5535714
##   Q 0.6103896 0.3896104
##   S 0.6630435 0.3369565
```
So, while 38.96% of passengers embarked from port Queenstown survived and 33.69% of those embarked from Southampton survived, 55.35% of passengers embarked from Cherbourg port survived.

# 4 Prediction

## 4.1 Imputation


```r
sum(is.na(train$Age))
```

```
## [1] 177
```
177 values are missing in the dataset. Simply ignoring the missing values can cause the model to overfit and can also result in bias analysis. 
 Here, we can use Median Imputation, which is the best type of imputing data where data is Missing At Random(MAR).


```r
#An example using median imputation
#train(x,y, preProcess = "medianImpute")
```
### Determining which variables to choose as Independent Variables

As we can see from the above graphs, Pclass, Age, Sex and Embarked have shown a distinct behaviour towards Survival rate.
So, I will choose these variable as the independent variables for my model.

## 4.2 Modelling 
 I am performing Random Forest, but in a different manner.
 The accuracy of the model can be improved by using a cross-validation. Cross- Validation makes folds of the wole dataset and then apply to the model several times and choosing the one with the best accuracy.


```r
#Choosing independent columns 
x <- train[,c("Age","Pclass","Sex","Embarked")]
#Choosing the dependent column
y <- train$Survived
```
Performing modelling

```r
#Set a random seed
set.seed(123)

#the method "ranger" here is a fast alternative of randomForest.
#trainControl is used to define cross-validation.
model<- train(x = x,y = y,preProcess = "medianImpute", method = "ranger", trControl = trainControl(method = "cv", number = 10))
```
Checking the model

```r
model
```

```
## Random Forest 
## 
## 891 samples
##   4 predictor
##   2 classes: '0', '1' 
## 
## Pre-processing: median imputation (1), ignore (3) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 801, 802, 802, 802, 803, 801, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.8249642  0.6065370
##   3     0.8036403  0.5652823
##   4     0.8013557  0.5735471
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```
## 4.3 Prediction !
This is the final point, performing prediction on the test dataset.

```r
# Predict using the test set
prediction <- predict(model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'rfSolution.csv', row.names = F)
```
