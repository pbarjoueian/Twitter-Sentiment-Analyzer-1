#set our working directory
wd = "C:/users/dsorvisto/Desktop/Twitter Sentiment Benchmarks"
setwd(wd)
getwd() == wd

library(twitteR)
library(tm)
library(wordcloud)
library(RColorBrewer)
require(sentR)

#import our data in csv file
#datacsv <- data.frame(read.csv(file.choose(),header = T))

consumer_key='ea6hTayVCxoF7epzS750lePPs' 
consumer_secret='RerbDO6W9xY1thwYlwwsvGfZ7L38JNge5NX3kmog2RnDkJTlEB'
alchemy_apikey='7db29e7116b60828b3e937aaf2f3503a1a34afb9'
access_token = 	'733739342133268480-xadGA115yUfbzupr7hQD6NmjpYZnlMO'
access_secret = '8Wcb3Hanb2wy829TdZVGjaz4aNMg3m38RLILh8R4EUnUc'

mach_tweets = searchTwitter("sportchek", n=500, lang="en")

mach_text = sapply(mach_tweets, function(x) x$getText())
# create a corpus
mach_corpus = Corpus(VectorSource(mach_text))

# create document term matrix applying some transformations
tdm = TermDocumentMatrix(mach_corpus,
                         control = list(removePunctuation = TRUE,
                                        stopwords = c("sportchek", stopwords("english")),
                                        removeNumbers = TRUE, tolower = FALSE))
# define tdm as matrix
m = as.matrix(tdm)
# get word counts in decreasing order
word_freqs = sort(rowSums(m), decreasing=TRUE) 
# create a data frame with words and their frequencies
dm <- data.frame(word=names(word_freqs), freq=word_freqs)
# plot wordcloud
wordcloud(dm$word, dm$freq, random.order=FALSE, colors=brewer.pal(8, "Dark2"))

#create classifier and export results to csv in real-time
cl <- classify.naivebayes(dm)
write.csv(cbind(mach_text,cl),file = "export.csv")