current results

LDA model trained using no_below=5 and no_above=0.4, during preprocessing also added bigrams.

training on about 50k articles (first 60 days). used gridsearchCV to find good parameters for an ExtraTreesClassifier (saved model too big to upload to github)

results on about 2,500 articles (days 60 through 70): f1 score 0.446.

distribution of documents is mostly conspiracy_pseudoscience, right_bias, right_center_bias, left_center_bias, and left_bias
