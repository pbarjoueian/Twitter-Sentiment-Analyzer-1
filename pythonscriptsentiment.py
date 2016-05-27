

#this list stores all training data loaded once and parsed
training = []

import re
def parse(tweet): 
    counter = 0 
    clean_string = ''
    delim = '"' # or whatever delimister used in Hive table
    for tw in tweet:
        if tw != delim and counter < 2 and ord(tw) < 128:
            clean_string += tw
        elif (tw == delim):
            counter += 1 
        else:
            break
            
    clean_string = clean_string.replace('@','').strip()
    clean_string = re.sub(r"http\S+", "", clean_string)
    clean_string = re.sub(r"https\S+", "", clean_string)
	clean_string = clean_string.replace("RT","",1)
    return clean_string.strip('\n').lower() #tweet cleaned with #hashtags still in place

import csv
def csv_loader(file_path,sentiment = 1): 
         with open(file_path) as csvfile:
             reader = csv.DictReader(csvfile)
             for row in reader:

                 training.append((parse(row['Text']),sentiment))


positive_path = 'positive1.csv' #make sure to label tweets as Text
negative_path = 'negative.csv'

pos = csv_loader(positive_path)
neg = csv_loader(negative_path, sentiment = 0)

#note to add a file to training simple call csv_loader and specify a sentiment


import random
random.shuffle(training) #shuffling is important to randomize training data to avoid overfitting


X,y = zip(*training) #list of lists of strings and list of list of labels (integers)


# Live Twitter Stream, Predict =============================
from textblob import TextBlob

def polarity_of_word(word):
    blob = TextBlob(word)
    return float(blob.polarity)

import re
def hash_tag(tweet):
    total = 0.0
    hash_set = set([re.sub(r"(\W+)$", "", j) for j in set([i for i in tweet.split() if i.startswith("#")])])
    if len(hash_set) > 0:
        max_length = max([len(s) for s in hash_set])
        longest_hash = [s for s in hash_set if len(s) == max_length ][0]
        return polarity_of_word(longest_hash) 
    else:
        return 0.0
  
from nltk.tokenize import word_tokenize

#features will always take a tweet as input and return a list so feature1+feature2 is another feature vector
def features(tweet): 
    happy = r"[:)]" #detect smilie faces
    sad = r"[:(]" #detect sad faces in tweets
    count_happy = len(re.findall(happy,tweet))
    count_sad = len(re.findall(sad,tweet))
    total = polarity_of_word(tweet) #text blob pre machine learning sentiment of entire tweet 
    positive_scores = []
    pos = 0.0 
    neg = 0.0
    for word in word_tokenize(tweet): #dependency on nltk replace by tokenizer 
        score = polarity_of_word(word) #or sentiment_dictionary.get(word,0)
        if score > 0.0: 
            pos = pos + score
            positive_scores.append(score) #positive_scores contains all positive tokens 
        if score < 0.0: 
            neg = neg + score
    #hash = hash_tag(tweet)
    normalize = len(tweet) + 1 #avoid division by zero
    if len(positive_scores) > 0: 
        maximum = max(positive_scores)
        last = positive_scores[-1]
        return [100*total,pos/normalize,neg/normalize,maximum/normalize,last/normalize,count_sad,count_happy] 
    else:
        return [100*total,pos/normalize,neg/normalize,0.0,0.0,count_sad,count_happy] 




X = list(map(features,X)) #for training with scikitlearn convert to feature vec representation, contains all training and test data


#=============================== Load Model and Train model


if __name__ == '__main__':
    import math
    from sklearn.svm import SVC
    import numpy as np
    from textblob import TextBlob 

    #implement a subjectivity classifier to filter out objective or neutral texts
    def textblob(tweet):
        blob = TextBlob(tweet)
        subj = blob.sentiment.subjectivity 
        polar = blob.sentiment.polarity
        return subj,polar



    #train model



    d = math.floor(len(X)*0.1) #defines split                   
    train_X = X[:-d]
    test_X = X[-d:]
    train_y = y[:-d]
    test_y =  y[-d:]
            
    X = np.array(train_X)
    y = np.array(train_y)
   

    clf = SVC(kernel='linear', C = 1, probability = True)
    model = clf.fit(X, y)
    
    #lets define a function that outputs the probability of each label 
    
    def predict_proba(xs):
        return clf.predict_proba(xs) #here xs is  a list of feature vectors 


    from nltk.corpus import stopwords
    
    stops = set(stopwords.words('english'))
    
           
    def clean(tweet):
        #figure out encoding and clean
        s = "" 
        for t in tweet:
         
            if int(t) < 128:
                s+=chr(t)
            else:
                s+=""
        clean = []        
        for token in s.split(' '):
            if token not in stops:
            
                token = re.sub(r'#\w+ ?', '',token)
                
                token = re.sub(r'[^0-9a-zA-Z\\/@+\-:,|#]+', '', token)
                token = re.sub(r'http\S+','',token)
                token = re.sub(r'https\S+','',token)
                token = re.sub(r'@\S+','',token)
                token = re.sub(r'\n+','',token)
				token = token.replace("RT","",1)
                token = token.replace('-',' ')
                token = token.replace('<',' ')
                token = token.replace('>',' ')
                token = token.replace('=',' ')
                token = token.replace('\\',' ')
                token = token.replace('/',' ')
                
          
 
                clean.append(token.lower())
        return str(' '.join(clean))
		
    predict_tweets = []
    with open('combined_englishonly.csv', 'rb') as binfile:
    
            temp = []
            readsall = binfile.readlines()
            for tweet in readsall:
                try:
                    temp.append(clean(tweet))
                    #print(clean(tweet))
                except (TypeError,ValueError):
                    continue 
            predict_tweets = temp

    
	
    def predict(tweet): #output function
        blob = textblob(tweet)
        neutral = blob[0] #this is a subjectivity classifier that gives an objectivity score for tweet 
        sentiment = blob[1] #this is the baseline sentiment without ML
        if  neutral > 0.15: #0.15 is threshhold so greater than this is neutral text
            feature_to_predict = np.ravel(np.array(features(tweet))).reshape(1,7)
            predict = predict_proba(feature_to_predict)
            #bench_mark = (predict[0][0] if sentiment > 0 else predict[0][1] #bench mark
        
        
            return [predict[0][1],sentiment,tweet]
        else:
      
            return ["NULL",sentiment,tweet] #dont run model

    prediction = [predict(tweet) for tweet in predict_tweets]
 
    with open('testexport.csv','w',newline='') as f:
        writer = csv.writer(f)
        for row in prediction:
            writer.writerow(row) 
    f.close()






