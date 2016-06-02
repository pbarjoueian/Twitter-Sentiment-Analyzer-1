 /* 
 This is a more sophisticated example of sentiment analyzer
 we need the following files:
 - polarity.csv
 - negative.csv (containing negative tweets for training and testing )
 - positive.csv (containing positive tweets for training and testing) */ 

 /* define a function to clean each tweet */
  def clean(s: String, stopwords: String) = s filterNot (stopwords contains _)
  stops = "@#()" //define characters to remove from Tweet
  
  /*load polarity dictionary into a Map structure */
  
  val lexicon = Source.fromFile("dayne2/polarity.csv").getLines.map(_.split(",")) // extract key-value pairs as a list with map and split
  val iter = lexicon.next // point an iterator to lexicon
  def polarity (word: String) : Map[String,Double] = lexicon.map(iter.zip(_).toMap) //create a dictionary of type Map[String,Double] 
  
  def features(tweet: String) : LabelledPoint = { 
    val tw = clean tweet 
	val tokens : List[Double] = tw.map( tweet => polarity getOrElse (tweet,0d) ).toList 
	
	// extract features
	val feat1 = tokens.length
	val feat2 = max tokens 
    val feat3 = tokens.last

    // return LabelledPoint 
    LabelledPoint(feat1,feat2,feat3) 	
	
  }  
  
  /* instantiate Spark context (not needed for running inside Spark shell */
    /* load positive and negative sentences from the dataset */
    /* let 1 - positive class, 0 - negative class */
    /* tokenize sentences and transform them into vector space model */
    val positiveData = sc.textFile("dayn2/positive.csv")
      .map { text => new LabeledPoint(1, features(text.split(" ")))}
    val negativeData = sc.textFile("dayne2/negative.csv")
      .map { text => new LabeledPoint(0, features(text.split(" ")))}
    /* split the data 60% for training, 40% for testing */
    val posSplits = positiveData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val negSplits = negativeData.randomSplit(Array(0.6, 0.4), seed = 11L)
    /* union train data with positive and negative sentences */
    val training = posSplits(0).union(negSplits(0))
    /* union test data with positive and negative sentences */
    val test = posSplits(1).union(negSplits(1))
    /* Multinomial Naive Bayesian classifier */
    val model = NaiveBayes.train(training)
    /* predict */
    val predictionAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }
    /* metrics */
    val metrics = new MulticlassMetrics(predictionAndLabels)
    /* output F1-measure for all labels (0 and 1, negative and positive) */
    metrics.labels.foreach( l => println(metrics.fMeasure(l)))