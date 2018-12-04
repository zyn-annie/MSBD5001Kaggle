
MSBD 5001 individual project
Description
This competition is about modeling the performance of computer programs. The dataset provided describes a few examples of running SGDClassifier in Python. The features of the dataset describes the SGDClassifier as well as the features used to generate the synthetic training data. The data to be analyzed is the training time of the SGDClassifier.
Key points and difficulties of the project
In this individual project, the main point is to find the basic linear relationship of running time and (max_iter* n_samples* n_features* n_classes/ n_jobs). In addition, too little data in the training data set makes the results of the model easy to overfit. 
Data preprocess and Feature engineering
Observing the training data, we can drop the id and random_state directly, which have little impact on running time. Besides, after checking some material on internet, I found the penalty, l1_ratio, alpha, n_clusters_per_class, flip_y, scale, n_informative, not the major factors of the time. At last, I chose penalty, l1_ratio, alpha, n_clusters_per_class, flip_y, scale, n_informative, max_iter, n_samples* n_features, (n_samples* n_features/ n_jobs), (n_classes* n_clusters_per_class) as feature. When the n_jobs is -1, I use the max values of n_jobs in dataset to replace them.
Model
I use SVM regression and XGBoost regression to predict the result, in the public test dataset, the xgboost model has better performance. The final parameter is shown below:
 
Actually, I also use cross validation to avoid overfitting, however, it did not perform well.
