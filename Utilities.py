import pandas as pd
import random
import seaborn as sn
from sklearn.utils import shuffle
import numpy as np
#import preprocessor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix
from matplotlib import pyplot as plt

def plot_confusion_matrix_custom(cm):
    

    df_cm = pd.DataFrame(cm, index = ["bot", "genuine"],
                      columns = ["bot", "genuine"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
    
def generate_data(p1=0.1, p3=0.04):

    file_route_genuine ="/drive/MyDrive/TextMiningProject/Datasets/datasets_full/genuine_accounts/tweets.csv"

    genuine_sample_ = pd.read_csv(
    file_route_genuine,
    header=None,
    usecols=[1],
    skiprows=lambda i: i>0 and random.random() > p1, 
    on_bad_lines='skip' # deprecated for some version,
      )

    genuine_sample = genuine_sample_[[1]]
    genuine_sample = genuine_sample.rename(columns={1:"text"})
    genuine_sample = genuine_sample.dropna()
    #genuine_sample["Label"] = ["genuine" for i in range(len(genuine_sample))]

    file_route_ssb2 = "Datasets/datasets_full/social_spambots_2/tweets.csv"
    ssb2_ = pd.read_csv(file_route_ssb2, encoding= 'unicode_escape')
    ssb2 = ssb2_[["text"]]
    #ssb2["Label"] = ["bot" for i in range(len(ssb2))] 
    ssb2 = ssb2.dropna()

    file_route_ssb3 = "Datasets/datasets_full/social_spambots_3/tweets.csv"
    
    ssb3_ = pd.read_csv(
    file_route_ssb3,
    header=None,
    usecols=[1],
    skiprows=lambda i: i>0 and random.random() > p3, 
    on_bad_lines='skip' ,# deprecated for some version,
    encoding= 'unicode_escape'
      )
    ssb3 = ssb3_[[1]]
    #ssb3["Label"] = ["bot" for i in range(len(ssb3))]
    ssb3 = ssb3.dropna()
    ssb3.columns = ssb3.iloc[0]
    ssb3 = ssb3.drop(0)

    return genuine_sample, ssb2, ssb3
    

def generate_train_test(genuine_sample, ssb2, ssb3):  


    ## SSB2 Bot Tweets

	# test
    ssb2_test = ssb2.sample(frac=0.3, replace=False, random_state=1234)
    ssb2_test["Label"] = pd.Series(["bot" for i in range(len(ssb2_test))])

	# train
    t = list(ssb2_test.index)
    ssb2_train = ssb2.loc[~ssb2.index.isin(t)]
    ssb2_train["Label"] = "bot"

    ## SSB3 Bot Tweets	

	# test	
    ssb3_test = ssb3.sample(frac=0.3, replace=False, random_state=1234)
    ssb3_test["Label"] = "bot" 

	# train
    t = list(ssb3_test.index)
    ssb3_train = ssb3.loc[~ssb3.index.isin(t)]
    ssb3_train["Label"] = "bot" 
    
    ## Genuine Human Tweets

	# test	
    genuine_test = genuine_sample.sample(frac=0.3, replace=False, random_state=1234)
    genuine_test["Label"] = "genuine"
    
	# train
    t = list(genuine_test.index)
    genuine_train = genuine_sample.loc[~genuine_sample.index.isin(t)]
    genuine_train["Label"] = "genuine"
	
    ## Final splits formation

	## SSB2 + Genuine (test)
    test_1 = pd.concat([ssb2_test, genuine_test], ignore_index=True)
    test_1 = shuffle(test_1)
    test_1 = test_1.dropna() 

	## SSB3 + Genuine (test)
    test_2 = pd.concat([ssb3_test, genuine_test], ignore_index=True)
    test_2 = shuffle(test_2)
    test_2 = test_2.dropna()     

	## SSB2 + Genuine (train)
    train_1 = pd.concat([ssb2_train, genuine_train], ignore_index=True)
    train_1 = shuffle(train_1)
    train_1 = train_1.dropna()         

	## SSB3 + Genuine (train)
    train_2 = pd.concat([ssb3_train, genuine_train], ignore_index=True)
    train_2 = shuffle(train_2)
    train_2 = train_2.dropna()
  


    return train_1, train_2, test_1, test_2


def downsample(data_df):
  count_smallest = data_df['Label'].value_counts().iloc[-1]
  df = pd.DataFrame()
  for i in data_df['Label'].value_counts().index:
    df = df.append(data_df[data_df["Label"] == i].sample(n=count_smallest, random_state=1234 ))
  df = df.sort_index(ascending=True)
  return df

def downsample_prop(data_df, prop=0.5):
  count_trim = round(prop*len(data_df))
  
  df = data_df.sample(n=count_trim, random_state=1234)
  
  return df

def report_values(train_1, train_2, test_1, test_2):

  values_df = pd.DataFrame(train_1["Label"].value_counts())
  values_df["SSB3 Train"] = train_2["Label"].value_counts()
  values_df["SSB2 Test"] = test_1["Label"].value_counts()
  values_df["SSB3 Test"] = test_2["Label"].value_counts()
  values_df.rename({"Label": "SSB2 Train"},  inplace=True, axis='columns' )

  return values_df


