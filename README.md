<h1 align="center">**Unsupervised Machine Learning**</h1>

Group Members:

- Clarissa Audrey Fabiola Kusnadi (2602118490)
- Jeffrey (2602118484)
- Priscilla Abigail Munthe (2602109883)



For this repository, we explored unsupervised machine learning techniques and applied four different algorithms to our dataset, namely K-Means, Gaussian Mixture Model, BIRCH, and K-Modes.



### <u>Clustering Algorithms</u>

> Clustering is the process of dividing the entire data into groups (also known as clusters) based on the patterns in the data.

#### 1. K-Means Clustering

**Description**:  K-Means clustering is a method for grouping observations into K clusters. Widely used in machine learning, it partitions data points into clusters based on similarity. Originally developed for signal processing, K-Means assigns each observation to the cluster with the nearest mean or centroid. The objective is to minimize the sum of squared distances between data points and their cluster centroids. This results in internally homogeneous clusters that are distinct from each other.

**Pros:** 

- Relatively simple to implement.
- Scales to large data sets.
-  Guarantees convergence.
- Can warm-start the positions of centroids.
- Easily adapts to new examples.
- Generalizes to clusters of different shapes and sizes, such as elliptical clusters.

**Cons:** 

- Choosing k manually.
- Being dependent on initial values.
- Clustering data of varying sizes and density.
- Clustering outliers.
- Scaling with number of dimensions.

**Code:**

```
# Applying KMeans clustering with the optimal number of clusters

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

clusters_kmeans = kmeans.fit_predict(scaled_data)

centroids = kmeans.cluster_centers_
```



#### 2. Gaussian Mixture Model Clustering

**Description**: The Gaussian Mixture Model (GMM) is a widely-used clustering method that represents data as a blend of Gaussian distributions. As a probabilistic clustering approach, GMM assigns a probability distribution to each cluster, enabling more flexible and precise clustering compared to alternative methods. Notably, GMM excels at modeling complex cluster shapes and accommodates overlapping clusters. Beyond clustering, GMM proves valuable for density estimation, allowing the estimation of the probability distribution for a given set of data points.

**Pros:**

- Has high flexibility of accomodating diverse distributions.
- Robust to outliers, efficiently handling multiple models.
- Efficient fitting, particularly with the EM algorithm.
- Adaptive to missing data, suitable for incomplete observations.
- Interpretable parameters for clear insights into data structure.

**Cons:**

- Strugle with convergence due to sensitivity to initial values.
- Assumption of normality.
- Selecting the right component count is challenging.
- Computationally demanding with high-dimensional data.
- Limited expressive power.

**Code:**

```
# Applying GMM clustering
gmm = GaussianMixture(n_components=4, random_state=42)
clusters_gmm = gmm.fit_predict(data_for_gmm)
```



#### 3. BIRCH Clustering

**Description:** BIRCH, also known as Balanced Iterative Reducing and Clustering using Hierarchies, is a better alternative to regular clustering algorithms like K-means. This clustering algorithm can cluster large datasets by generating a small and compact summary while retaining as much information as possible, then the algorithm will use this smaller summary instead. This clustering algorithm can even be used to complement other algorithms by making a summary of the dataset that other algorithms will then use.

**Pros:**

- Scalability 
- Memory efficiency
- Incremental processing
- Noise handling
- Balanced clusters 

**Cons:**

- Sensitive to parameters
- Limited to spherical clusters
- Requires sufficient memory
- Hierarchical nature

**Code:**

```
# Initialize Birch with desired parameters
birch = Birch(n_clusters=4)  # You can set the number of clusters as needed
```



#### 4. K-Modes Clustering

**Description:** This last clustering method is mainly made for handling categorical data, using modes instead of means or medians to represent cluster centers. Modes in this context represent the most frequently occurring values for categorical variables within each cluster. The algorithm employs dissimilarity measures designed for categorical data, such as the Jaccard distance, to assess the similarity between data points. 

**Pros:** 

- Suitable for categorical data
- Interpretability
- Handles mixed data types
- Robust to outliers
- Simple and intuitive

**Cons:**

- Limited application to numerical data
- Dependence on initial centroids
- Sensitivity to number of clusters
- Computational complexity
- Non-euclidean dissimilarity measures

**Code:**

```
km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)  # You can set the number of clusters as needed
clusters_kmodes = km.fit_predict(data_kmodes)
```



### <u>Results and Evaluation</u>

#### Results

In this section, visual representations of the clustering results are provided for each algorithm:

- K-Means Clustering![k-means_result](/Assets/k-means_result.png)

- Gaussian Mixture Model Clustering![gaussian_result](/Assets/gaussian_result.png)

  

- BIRCH Clustering![BIRCH_result](/Assets/BIRCH_result.png)

- K-Modes Clustering

![k-modes_result](/Assets/k-modes_result.png)

#### Evaluation

In this section, we assess the performance of different clustering methods using the Silhouette Score, a metric that measures how well-defined the clusters are within the data. Each clustering method is evaluated independently:

| Method                            | Silhouette Score   |
| :-------------------------------- | :----------------- |
| K-Means Clustering                | 0.6983973143019458 |
| Gaussian Mixture Model Clustering | 0.4996946747441537 |
| BIRCH Clustering                  | 0.9537703651625458 |
| K-Modes Clustering                | 0.5435766047259253 |

Among the clustering algorithms assessed through silhouette scores, Birch stands out with a high score of 0.9538, indicating well-formed and compact clusters. KMeans also performs well, achieving a score of 0.6984, suggesting clear cluster separation. In contrast, KModes exhibits moderate clustering quality with a score of 0.5436, while GMM shows the lowest score at 0.4997, indicating less distinct clusters. So in conclusion, Birch appears to be the most effective for creating clear and cohesive clusters in the given data, followed by KMeans, while KModes and GMM exhibit quite a lower performance in terms of clustering quality.



