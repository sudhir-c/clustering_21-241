We will attempt to solve the clustering problem on 3 datasets by comparing 3 approaches.

    - spectral clustering using an unnormalized Laplacian graph, followed by k-means clustering on the output of the spectral clustering
    - spectral clustering using an normalized Laplacian graph, followed by k-means clustering on the output of the spectral clustering (Ng-Jordan-Weiss algorithm)
    - the k-means clustering algorithm

We will contrast k-means as an approach with spectral clustering overall, and also discuss the theoretical effects of using an unnormalized vs normalized Laplacian. Furthermore, for the spectral clustering approaches, we will discuss the effect of our choice of sigma for our similarity function.  
