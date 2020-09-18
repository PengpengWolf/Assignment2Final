1. The optimal number of cluster is determined in line 270 in the code file assignment2.py. Since my code is not very efficient, 
I did not use elbow method otherwise it would take a long time. I just chose values from 1 to 15 and compared all the efficiencies.

2. Every value in the dataset is tagged twice. The first tag is added when the data is generated. The second is when defining the clusters in kmeans according centroids.
In the end, the two tags are compared. If they are the same, the single data is correctly classified. The final efficiency is calculated by correct number dividing total test number.
I list the efficiency for every state and the total effiency for all the states. Unfortunately, I have no idea how add a video or GUI to show that. Is there any example for me to learn from?