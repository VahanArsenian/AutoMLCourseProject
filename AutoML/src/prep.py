# ----------------- import the necessary libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#from nltk.tokenize import TweetTokenizer
from nltk.tokenize import  word_tokenize
import pandas as pd


# -------------------- preprocess image data
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    length, height, depth = image.shape
    v = image.reshape((length * height * depth, 1))    
    
    return v

def images2vector(images):
	"""

	"""


# -------------------- preprocess text data

# an example of pdf loader
#import textract
#text = textract.process("C:\\Users\\Tigran\\Desktop\\AutoML\\notebooks\\monetary.pdf")

def text_prep(text):
    """Process string function.
    Input:
        text: a string containing a text
    Output:
        texts_clean: a list of words containing the processed text

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # remove stock market tickers like $GE
    #tweet = re.sub(r'\$\w*', '', tweet)
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    
    # tokenize text
    text_tokens = word_tokenize(text)

    texts_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            texts_clean.append(stem_word)

    return texts_clean

# initialize the TfIdf vectorizer
vectorizer = TfidfVectorizer(tokenizer = text_prep)

# for testing the vectorizer
#x = vectorizer.fit_transform(txt)
#tfidf_tokens = vectorizer.get_feature_names()
#df_tfidfvect = pd.DataFrame(data = x.toarray(), columns = tfidf_tokens)
#print("\nTD-IDF Vectorizer\n")
#df_tfidfvect


# -------------------- preprocess tabular data, auto_prep(...)
# for engineering the dates
# parse dates should be specified in pandas read csv
def add_datepart(df, fldname):
    fld = df[fldname]
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
              'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 
              'Is_year_start'):
        df[fldname+n] = getattr(fld.dt, n.lower())
    df.drop(fldname, axis = 1, inplace = True)

#l = len(dates)
#for i in range(l):
#    idx = dates[i]
#    fldname = df.columns[idx]
#    add_datepart(df, fldname)

# ---------------
def add_date_related_features(data, date_field_name):
    df_dt = pd.to_datetime(df[date_field_name])
    for cst_ft_name in ('year', 'month', 'week', 'day', 'dayofweek', 'dayofyear',
              'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end',
              'is_year_start'):
        df[date_field_name + "_" + n] = getattr(df_dt.dt, cst_ft_name)
    df.drop(columns = [date_field_name], axis = 1, inplace = True)

def std_scaler(scaler, col):
    """
    Fits sklearn's standard scaler
    scaler:
        scaler object to be fitted
    col:
        column of interest
    """
    return scaler.fit_transform(col.values.reshape(-1,1))

def freq_code(db):
    """
    Frequency encoder
    db:
        DataFrame
    """
    for i in db.dtypes.loc[db.dtypes == 'object'].index:
        encoding = len(db.groupby(i).size())/len(db)
        if encoding < 0.8:
            encoding = db.groupby(i).size()
            encoding = encoding/len(db)
            db[i + '_freq_enc'] = db[i].map(encoding)

def mean_std_rows(df):
    """
    Computes first and second central moments
    df:
        DataFrame
    """
    columns = []
    for cl in df.columns:
        cl_loc = df.loc[:,cl]
        if is_numeric_dtype(cl_loc):
            columns.append(cl)
    df["mean"] = df[columns].apply(np.mean, axis = 1)
    df["std"] = df[columns].apply(np.std, axis = 1)


def autoprep(PATH, y, dates=None, scalevars=False, testsize=0.3, categorical=False):
    """
    Applies all our auto prerprocessing steps
    file_url:
        URL location of our dataframe (csv)
    cl_index:
        Target columns index
    dates:
        Columns which contains dates
    scalevars:
        Flag which defines usage of scaling
    test_size:
        Proportion of test data
    """
    data = pd.read_csv(file_url)
    column = data.iloc[:, cl_index - 1]
    X = data.drop(data.columns[cl_index - 1], axis=1)
    y = column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    for colname in X_train.columns:
        coltrain = X_train.loc[:, colname].copy()
        namask = coltrain.isna()
        if dates and colname in dates:
            add_datepart(X_train, colname)
        elif len(coltrain.unique()) <= 50:
            coltrain.loc[~coltrain.isna()] = label_encoder.fit_transform(coltrain.loc[~coltrain.isna()])
            coltrain.loc[coltrain.isna()] = -1
            X_train.loc[:, colname] = coltrain
        elif is_numeric_dtype(coltrain):
            coltrain = coltrain.fillna(coltrain.median())
            if scalevars:
                coltrain = StandardScale(scaler, coltrain)
            X_train.loc[:, colname] = coltrain
        if namask.sum() > 0:
            X_train[colname + "_isna"] = namask
            X_train.loc[:, colname] = coltrain
    freq_code(X_train)
    mean_std_rows(X_train)
    return ((X_train, y_train), (X_test, y_test))

