This repository holds an approach to the SemEval2012 competition of task 6, Semantic Textual Similarity. Please note that I did not participate in the original competition and that this assignment is academically driven.

The details on the implementation are displayed in a Jupyter Notebook.

The set of features used in the end models are displayed below in a correlation matrix  

![image](https://user-images.githubusercontent.com/8356912/71176102-54552c80-2269-11ea-95d4-30b7d3f4ed8e.png)

In case you want to re-compute the features you need to install CoreNLP and configure it as a Server. Additionally, you need to download the Glove 300 model of your preference (download it [here](https://nlp.stanford.edu/projects/glove/)) and reference it in the features.py file.

The training dataset was obtained from [this](https://github.com/anantm95/Semantic-Textual-Similarity) repository. The high accuracy obtained in this implementation relies on the fact that this augmented training dataset encapsulates the training sets from the same competition from the year 2012 to 2017. 

The resultant models' performance is displayed in the Figure below. Each model uses a subset of the _relevant features_ obtained by hand tunning or recursive feature elimination (or both, more details in the notebook).  

![image](https://user-images.githubusercontent.com/8356912/71176423-291f0d00-226a-11ea-942c-e064d1873279.png)

_(BoW): Stands for models using as one of its features the outcome of a regressor model trained only with BoW tf/idf embeddings_
