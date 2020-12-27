# AutoMLCourseProject
The aim of this project is to provide a simple AutoML tool for daily tasks. 
The code consists of 3 main parts.


## Processing Image Data
Flattens the given image from [width, height, depth] to [width*height*depth, 1]

## Processing Text Data
Followings are applied in the given order:
1. Stemming: Extract stem from a word (PorterStemmer is used, which is a rule based)
2. Dropping english stopwords
3. Dropping numbers and old style retweets
4. Dropping Hyperlinks 
5. Tokenization (Tree bank word tokenizer is used)

## Processing Tabular Data
Followings are applied in the given order:
1. Train/test split by testsize proportion
2. Extra features for missing values 
  a. A boolean column indicating the basence of value
3. If the feautre is categorical label encode the values and create new class for missing values with value -1
4. If the feature is numeric, replace missing values with the median
5. Scale numerica values via StandardScaler (controlled by scalevars parameter)
6. Feature extracting from dates (e.g. day of a week, year, month, and etc.)
7. Adds mean and std of columns as additional features 
