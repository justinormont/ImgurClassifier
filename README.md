# ImgurClassifier
This project classifies [imgur](https://imgur.com/) posts as the likelihood of making it to the front page of imgur.

Dataset is multimodal, consisting of (3x) images, various text, categorical, and numeric features.

## Purpose
Primary purpose is to demonstrate advanced feature engineering techniques in ML.NET.

Techniques demonstrated:
* [Model stacking](https://en.wikipedia.org/wiki/Model_stacking)
  * k-means cluster distance featurizer
  * Text target encoding (stacked linear model on text ngrams)
  * Alt-label -- stacking of a regression model towards an alternative label
  * Model stacking using only subset of features (images, text, ...)
  * Stacking of AutoML models
* Multi-threaded parallel model creation
* Expression transform to create a `Weight` column
* Image featurization
* Label rotation to allow a regression model to train on a multi-class dataset
* String statistics (length, number of vowels, word count, ...) using a Custom Mapper
* Explainability -- feature importance from model weights via training a proxy model
