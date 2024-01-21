# EmailSift: AI-Driven Email Clustering and Analysis

## Overview
EmailSift is an unsupervised machine learning project which seeks to organize and interpret emails according to the content of their body. Advanced clustering techniques, using a K-Means approach, helps the model categorize emails into meaningful groups, enabling easier navigation and interpretation of email datasets.

## Features
- **Email Parsing:** Extracts key components of emails including sender, recipient, and body.
- **TF-IDF Vectorization:** Converts email body content into numerical format for machine learning processing.
- **K-Means Clustering:** Groups emails into clusters based on content similarity.
- **Performance Evaluation:** Uses metrics like Silhouette Score, Separation, Cohesion, and Centroid distances to assess the quality of clustering.
- **Scalability:** Handles large datasets efficiently by splitting and processing in manageable parts.

## Installation
Before you begin, ensure that you have Python installed on your system. Clone the repository to your local machine:

```console
git clone https://github.com/aidantambling/AI-Driven-Email-Clustering-and-Analysis.git
cd AI-Driven-Email-Clustering-and-Analysis
```

Install the required dependencies:

```console
pip install -r requirements.txt
```

## Usage
To run the email clustering script, navigate to the project directory and execute:

```console
python email_clustering.py
```

## Configuration
- Modify the `config.py` file to set parameters like the number of clusters, vectorization settings, etc.
- Email data should be placed in the `data` directory in CSV format.

## Acknowledgements
- This project was collaborated on by Sharan Majumder of the University of Florida
- This project was inspired by the need to manage large volumes of email data efficiently.

## Contact
For any queries regarding this project, please contact:

- Aidan Tambling - atambling@ufl.edu
- Project Link: https://github.com/aidantambling/AI-Driven-Email-Clustering-and-Analysis
