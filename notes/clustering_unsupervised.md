# Clustering & Unsupervised Learning

Unsupervised learning uncovers structure without labels.  Distance metrics (Euclidean, Manhattan, cosine) and feature scaling set the stage.  **k‑means** partitions points into k clusters by minimizing within‑cluster variance; we discuss initialization strategies (k‑means++), convergence behaviour and how to interpret inertia.  Hierarchical clustering builds dendrograms using agglomerative or divisive approaches; linkage criteria (single, complete, average) lead to different shapes.

Density‑based clustering such as **DBSCAN** finds arbitrarily shaped clusters and identifies noise by examining neighbourhood density.  Students compare these algorithms using silhouette scores and Davies–Bouldin indices to choose k or epsilon.  We emphasize that clustering is exploratory: different algorithms capture different notions of similarity, and results require domain interpretation.
