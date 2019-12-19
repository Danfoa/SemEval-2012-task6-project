This repository holds an approach to the SemEval2012 competition of task 6, Semantic Textual Similarity.

The code is displayed on the main Jupyter Notebook, while the measurements of simmilarity and features extraction are coded into traditional .py files.

The set of features used in the end models are displayed below in a correlation matrix  

![image](https://user-images.githubusercontent.com/8356912/71176102-54552c80-2269-11ea-95d4-30b7d3f4ed8e.png)

Each feature implementation can be found in the repository, except for the Glove 300 model used, which can be downloaded [here](https://nlp.stanford.edu/projects/glove/)

The training dataset was obtained from [this](https://github.com/anantm95/Semantic-Textual-Similarity) repository which combines the trainins sets from the same competition task from the years 2012 to 2017.

The results can be summarized by the following picture. Each model uses a subset of the _relevant features_ obtained by hand tunning or recursive feature elimination (or both, more details in the notebook).  

![image](https://user-images.githubusercontent.com/8356912/71176423-291f0d00-226a-11ea-942c-e064d1873279.png)

_(BoW): Stands for models using as one of its features the outcome of a regressor model trained only with BoW Tf/idf embeddings_
