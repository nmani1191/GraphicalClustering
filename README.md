# GraphicalClustering
Tool to cluster IT support tickets based on its description/resolution remarks

## Defining the area of scope
When business grows, it’s dependency with IT systems also grows proportionally and monitoring the health of this IT systems are the key to ensure smooth functioning of business and IT. This IT systems are continuously monitored and managed by the IT service providers with group of resolvers. Any problem in the system will be reported as Trouble ticket to the resolvers and they take required corrective action to maintain good health of IT systems. The system problems are reported to the resolvers in two ways:

**System-generated issues:** Components such as business applications, processes, CPU, disk, and network interfaces are monitored to detect anomalies. The monitoring tools capture and report any abnormal behavior of system components in the form of alerts.

**User-generated issues:** The users of IT systems report problems through calls, emails, or chats.

Ticket contains various information about the reported problem and all of them can be grouped into any one of the below
* **Datetime fields** Like reported time, response time, resolved time, etc
* **Categorical fields** which provide generic grouping of the issue like impacted component, system-of-origin, cause code, CTI, etc
* **Text fields** like Issue Description, Notes, Summary, etc

Compared to categorical fields, Issue Description will contain more information about the problem, and it can be very effective in identifying the issues occurring in the IT system. It provides crucial information to better understand and control the IT operations. In case of system-generated tickets, this description is automatically generated by the monitoring and alerting tools. This description is often structured and is based on how the alerting tools have been configured to report a problem. However, in case of user-generated tickets, this description contains free-form text written by users. This description is unstructured and contains variations, ill-formed sentences, and spelling and grammar mistakes.

This article is about how effectively we can extract issue details from the system generated tickets because around 60–70% of the IT system tickets are system generated only.

## Analytics driven approach to mine textual descriptions of tickets to extract issues

Before getting into the proposed approach, let’s rewind currently available approaches

* The most common approach used to infer issues is the manual intuition-driven approach. The service providers infer the issue information either from the monthly reports manually produced by the teams or by talking to the domain experts. This approach carries the risk of being incomplete, inaccurate, and of variable quality.

* Another most common approach used is the n-gram extraction approach. Extracting the most common repeated n-grams from the tickets and group the tickets based on the same. Even though it reduces the manual intervention as like above approach, it creates only generic group and not able to take actions like automations or identifying the permanent resolution because the groups contain more than one similar issue which yet to be manually identified.

There is an inherent difference in the structure and heterogeneity in the descriptions of system-generated and user-generated tickets. The system-generated tickets have structured and consistent descriptions. So, we can very well apply clustering-based technique for system-generated tickets.

The system-generated tickets still contain variation based on configuration of the monitoring tools for different applications in the system. We address those variation by clustering the tickets based on similarity. Clustering helps grouping similar descriptions together and assign dissimilar descriptions to separate groups.

Consider two descriptions as

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

1. Compute Dice similarity between every pair of clean description.
2. Construct a similarity graph of clean descriptions in which nodes are clean descriptions.
3. Draw an edge between two clean descriptions if they are similar. Consider two clean descriptions similar if the similarity coefficient between them is greater than a predefined threshold threshold similarity.
4. Cluster clean descriptions by applying graph clustering on the similarity graph of clean descriptions. Various graph clustering techniques can be used for clustering such as cliques, connected components, graph partitioning, graph cuts, etc. We have used cliques to identify clusters of clean descriptions.
