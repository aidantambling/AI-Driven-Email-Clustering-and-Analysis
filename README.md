# EmailSift: AI-Driven Email Clustering and Analysis

## Overview
EmailSift is an unsupervised machine learning project which seeks to organize and interpret emails according to the content of their body. Advanced clustering techniques, using a K-Means approach, help the model categorize emails into meaningful groups, enabling easier navigation and interpretation of email datasets.

## Motivation
With the increasing reliance on email as a primary communication medium, efficiently sorting and categorizing emails according to content is becoming essential. This project aims to provide a robust solution to managing large volumes of email data.

## Dataset
The project utilizes the Enron Corpus, a comprehensive dataset of over 500,000 emails from the early 2000s. The dataset was chosen due to its diversity and volume, providing a realistic environment for email clustering.

## Features
- **Email Parsing:** Extracts key components of emails including sender, recipient, and body.
- **TF-IDF Vectorization:** Converts email body content into numerical format for machine learning processing.
- **K-Means Clustering:** Groups emails into clusters based on content similarity. The model uses 7 clusters.
- **Performance Evaluation:** Uses metrics like Silhouette Score, Separation, Cohesion, and Centroid distances to assess the quality of clustering.
- **Scalability:** Handles large datasets efficiently by splitting and processing in manageable parts.

![image](https://github.com/aidantambling/AI-Driven-Email-Clustering-and-Analysis/assets/101668617/dc1b7e2b-876d-4040-a592-e69c8817a526)

Sample email cluster shown above. This cluster seems to represent system-generated emails.

## Performance and Evaluation
The model showed distinct clustering of emails with a Silhouette Score of 0.03 for both training and testing datasets. While clusters are distinguishable, they are in close proximity, indicating room for further refinement.

## Limitations and Future Work
The model's current limitation is the reliance on a corporate email dataset, which may not generalize well to other types of email content. Future work could include diversifying the dataset and exploring more granular clustering techniques.

## Installation
Python should be installed on your system and up-to-date. You can check Python's status on your machine:
```console
py
```

Next, clone the repository to your local machine:

```console
git clone https://github.com/aidantambling/AI-Driven-Email-Clustering-and-Analysis.git
cd AI-Driven-Email-Clustering-and-Analysis
```

Install the model's dependencies for it to operate. For ease of use, they are contained in requirements.txt. Install them with:

```console
pip install -r requirements.txt
```

## Usage
To run the email clustering script, navigate to the project directory and execute:

```console
python main.py
```

## Acknowledgements
- This project was developed in collaboration with Sharan Majumder of the University of Florida.
- This project was inspired by the need to manage large volumes of email data efficiently.

## Contact
For any queries regarding this project, please contact:

- Aidan Tambling - atambling@ufl.edu
- Project Link: https://github.com/aidantambling/AI-Driven-Email-Clustering-and-Analysis

## Final Project Report
For a detailed overview of the project, including the motivation, implementation specifics, and comprehensive performance analysis, refer to our [Final Project Report](tambling_majumder_report.pdf)
