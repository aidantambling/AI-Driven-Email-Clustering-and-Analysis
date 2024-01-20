import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# # function to split large .csv file (500,000 emails) into small .csv file (10,000 emails)
# # function was used to create "emails_sample.csv" (training) and "emails_testSample.csv" (testing)
# # function is commented out so as to not accidentally tamper with files or create new ones

# large_file = 'emails.csv'
# sample_size = 10000
#
# # read the csv and create a random sample of 10,000 elements
# df = pd.read_csv(large_file)
# sample_df = df.sample(n=sample_size)
#
# # write the 10,000 elements to a new .csv file
# sample_file = 'emails_testSample.csv'
# sample_df.to_csv(sample_file, index=False)

emails = pd.read_csv('emails_sample.csv')

# function takes an email message and extracts the 'from' 'to' and 'body' components
def parse_email(message):
    # some data we seek to extract (from, to, body)
    headers = ['from', 'to']
    email_data = {'from': '', 'to': '', 'body': ''}
    body_lines = []
    in_headers = True  # Flag to track whether we are still reading headers

    # read the email line by line
    for line in message.split('\n'):
        # Check for the end of headers
        if in_headers and line.strip() == '':  # blank line indicates the headers have ended
            in_headers = False
            continue
        # identify the from, to data
        if in_headers and ':' in line:
            key, _, value = line.partition(':')
            if key.lower() in headers:
                email_data[
                    key.lower()] = value.strip()  # if the from / to field is found, add that to our email_data dict
        elif not in_headers:
            body_lines.append(line.strip())  # for a generic line, add that to our message body (but only if we've exited the header)

    email_data['body'] = ' '.join(body_lines)  # join the message body into a single element for our email_data dict
    return email_data


# process the pd dataframe of emails into parsed emails
parsed = [parse_email(msg) for msg in emails['message']]  # for each email message, parse the email. 'parsed' is a list of dicts
parsed_emails = {'from': [], 'to': [], 'body': []}  # we now transpose 'parsed' into a dict of lists
for email in parsed:
    parsed_emails['from'].append(email['from'])
    parsed_emails['to'].append(email['to'])
    parsed_emails['body'].append(email['body'])

# print sample email
print("Sample email body:", parsed_emails['body'][1])
print("Sample 'to' field:", parsed_emails['to'][1])
print("Sample 'from' field:", parsed_emails['from'][1])

emails_body = parsed_emails['body']  # extract the 'body' list from our 'parsed_emails' dict

# vectorization of the email bodies
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf_vectorizer.fit_transform(emails_body)

# k-means clustering
km = KMeans(n_clusters = 7, random_state=42)
km.fit(X)  # fit the k-means model to the tf-idf matrix

# inertias = []
# silhouette_scores = []
# K = range(2, 11)
#
# for k in K:
#     kmeans_model = KMeans(n_clusters=k, random_state=42)
#     kmeans_model.fit(X)  # replace 'your_data' with your dataset
#     inertias.append(kmeans_model.inertia_)
#
# plt.figure(figsize=(8, 6))
# plt.plot(K, inertias, 'bo-')
# plt.xlabel('Number of Clusters, k')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show()
#
# for k in K:
#     kmeans_model = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans_model.fit_predict(X)
#     silhouette_scores.append(silhouette_score(X, cluster_labels))
#
# # Plotting the silhouette scores
# plt.figure(figsize=(8, 6))
# plt.plot(K, silhouette_scores, 'bo-')
# plt.xlabel('Number of Clusters, k')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Scores for Different k')
# plt.show()


# link the cluster assignments with the emails
clusters = km.labels_.tolist()
emails_df = pd.DataFrame(parsed_emails)  # new dataframe of 'parsed_emails'
emails_df['cluster'] = clusters  # create new column for an email in the dataframe representing its assigned cluster

# Display sample emails from each cluster
num_emails_to_display = 3
for cluster_num in range(7):
    print(f"Cluster {cluster_num} sample emails:")
    cluster_emails = emails_df[emails_df['cluster'] == cluster_num]
    for _, row in cluster_emails.head(num_emails_to_display).iterrows():
        print(f"Email:\nFrom: {row['from']}\nTo: {row['to']}\nBody:\n{row['body']}\n")
    print("\n")

# read in testing data
new_emails = pd.read_csv('emails_testSample.csv')

# process the testing emails similarly to the training emails
new_parsed = [parse_email(msg) for msg in new_emails['message']]  # for each email message, parse the email. 'new_parsed' is a list of dicts
new_parsed_emails = {'from': [], 'to': [], 'body': []}  # we now transpose 'new_parsed' into a dict of lists
for new_email in new_parsed:
    new_parsed_emails['from'].append(new_email['from'])
    new_parsed_emails['to'].append(new_email['to'])
    new_parsed_emails['body'].append(new_email['body'])

new_emails_df = pd.DataFrame(new_parsed_emails)  # new dataframe of 'parsed_emails'

# vectorize the emails
new_emails_vectorized = tfidf_vectorizer.transform(new_emails_df['body'])

# predict the clusters and link them to the dataframe of new emails
predicted_clusters = km.predict(new_emails_vectorized)
new_emails_df['predicted_cluster'] = predicted_clusters

# Display sample emails from each predicted cluster
num_emails_to_display = 3
for cluster_num in range(7):
    print(f"Cluster {cluster_num} sample emails:")
    cluster_emails = new_emails_df[new_emails_df['predicted_cluster'] == cluster_num]
    for _, row in cluster_emails.head(num_emails_to_display).iterrows():
        print(f"Email:\nFrom: {row['from']}\nTo: {row['to']}\nBody:\n{row['body']}\n")
    print("\n")

# Display performance data
score = silhouette_score(X, clusters)
print("Silhouette Score (Training emails): ", score)

score = silhouette_score(new_emails_vectorized, predicted_clusters)
print("Silhouette Score (Testing emails): ", score)

cohesion = km.inertia_
total_mean = np.mean(X, axis=0)
separation = sum(np.linalg.norm(km.cluster_centers_ - total_mean, axis=1)**2)

centroids = km.cluster_centers_
centroid_distances = pairwise_distances(centroids)

print("Separation: ", separation)
print("Cohesion: ", cohesion)
print("Centroid distances: ", centroid_distances)


