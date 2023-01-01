# GraphicalClustering
Tool to cluster IT support tickets based on its description/resolution remarks

For example, lets consider two descriptions as

* BMC portal alert on d-hw6ttk1-lvmh: /var for filesystem full
* BMC portal critical alert on d-hw6ttk1-lvmh/var for filesystem full

These two descriptions are same except the word critical present in the second description. We group such similar descriptions into one cluster by computing similarity between two descriptions.

Some of the approaches to compute similarity between two descriptions are Jaccard coefficient, Dice coefficient, etc. Dice coefficient gives twice the weight to common elements. Since we emphasize on commonality, we use Dice coefficient to compute similarity between two descriptions.

Let A and B be sets of words in two descriptions. Dice similarity, D, between A and B is defined as follows:

**D = (2 ∗ |A ∩ B|)/ (|A| + |B|)**

For example, if
* A = BMC portal alert on d-hw6ttk1-lvmh: /var for filesystem full
* B = BMC portal critical alert on d-hw6ttk1-lvmh: /var for filesystem full
* then |A| = 9, |B| = 10, |A ∩ B| = 9 and D = (2∗9)/ (9+10) = 0.947.

Below are the logical steps to perform Graphical clustering after identifying the similarity scores
![alt text](https://github.com/nmani1191/GraphicalClustering/blob/main/Graphical_Clustering_flow.jpg?raw=true)


1. Compute Dice similarity between every pair of clean description.
2. Construct a similarity graph of clean descriptions in which nodes are clean descriptions.
3. Draw an edge between two clean descriptions if they are similar. Consider two clean descriptions similar if the similarity coefficient between them is greater than a predefined threshold threshold similarity.
4. Cluster clean descriptions by applying graph clustering on the similarity graph of clean descriptions. Various graph clustering techniques can be used for clustering such as cliques, connected components, graph partitioning, graph cuts, etc. We have used cliques to identify clusters of clean descriptions.

For more information about the how we evalued from the level of pseudo code into the full python package, kindly go through below articles

* https://medium.com/@nmani.1191/implementing-graphical-clustering-algorithm-from-research-paper-in-python-part-2-developing-the-ca6c1a1c8d84
* https://medium.com/@nmani.1191/how-we-addressed-the-challenges-faced-with-the-first-version-of-graphical-clustering-mvp-part-3-89d1420ec8c0

To install the module
```
pip install Graphical-Clustering
```

Read sample IT Tickets with description in order to cluster the same
```
import pandas as pd
input_tck_data = pd.read_csv('data/sample_data.csv')
input_text = input_tck_data['Issue Description']
```

Import the Graphical Clustering class and cluester the input data
```
from GraphicalClustering.graph_cluster import GraphicalClustering
gc_model = GraphicalClustering(input_text,2)
gc_model.generate_graph_clusters(5)
```
We can view the generated clusters like below
![alt text](https://github.com/nmani1191/GraphicalClustering/blob/main/output.jpg?raw=true)

We can save the generated cluster model for predicting future tickets as well like below
```
import joblib
joblib.dump(gc_model, 'gc_model_v1.pkl')
```

We can predict the new texts using the model saved like below
```
loaded_gc_model = joblib.load('gc_model_v1.pkl')
result = loaded_gc_model.predict(input_text[0:5]) #For testing here I passed same input again
```

Pedicted outputs will be like below
![alt text](https://github.com/nmani1191/GraphicalClustering/blob/main/prediction_output.jpg?raw=true)
