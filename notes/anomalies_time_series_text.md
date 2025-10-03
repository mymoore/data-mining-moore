# Anomalies, Time Series & Text Mining

Detecting anomalies requires unsupervised or semi‑supervised methods.  We explore robust statistics for z‑score filtering, the Local Outlier Factor (LOF) for density‑based anomaly detection, and **Isolation Forests**, which isolate anomalies via random partitioning.  Evaluating anomaly detectors is challenging without labels; students use precision‑at‑k and unsupervised scores like LOF.

Time series present temporal dependencies.  We decompose series into trend, seasonal and residual components; test for stationarity; and introduce **ARIMA** models for linear forecasting.  Tree‑based ensembles and gradient boosting are applied to tabular features extracted from lags and rolling windows, illustrating modern approaches like **Prophet** and `lightgbm` on time series.  Cross‑validation strategies respect temporal ordering (rolling and expanding windows).

In **text mining**, raw documents are cleaned by tokenization, normalization, and removal of stopwords.  Bag‑of‑words and TF‑IDF representations feed linear classifiers such as logistic regression and Naïve Bayes.  We discuss n‑gram features, word embeddings (Word2Vec, GloVe) and latent Dirichlet allocation (LDA) for topic modeling.  Ethics of text analytics, such as bias and privacy, are highlighted.
