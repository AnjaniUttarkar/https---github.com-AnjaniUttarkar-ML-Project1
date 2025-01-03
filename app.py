from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from googlesearch import search
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import re
from urllib.parse import urlparse
from tld import get_tld

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))


df = pd.read_csv("malicious_phish.csv")
df_phish = df[df.type == 'phishing']
df_malware = df[df.type == 'malware']
df_deface = df[df.type == 'defacement']
df_benign = df[df.type == 'benign']

# Feature extraction functions (make sure these are well-defined and capture phishing patterns)
def having_ip_address(url):
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.' 
        r'([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'  # IPv4
        r'((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)'  # IPv4 in hex
        r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    return 1 if match else 0


# Check if URL is abnormal (contains suspicious or unusual patterns)
def abnormal_url(url):
    abnormal_patterns = ['@', 'www.', '.com', '.org', '.net', '.info', '.xyz', 'http://', 'https://']
    if any(pattern in url for pattern in abnormal_patterns):
        return 1
    return 0

# Count the number of dots (.) in the URL
def count_dot(url):
    return url.count('.')

# Count the number of "www" in the URL
def count_www(url):
    return url.lower().count('www')

# Count the number of "@" symbols in the URL
def count_atrate(url):
    return url.count('@')

# Count the number of directories ("/") in the URL
def no_of_dir(url):
    return url.count('/')

# Count the number of embedded domains (e.g., suspicious second-level domains)
def no_of_embed(url):
    domain_count = len(re.findall(r'\.[a-z]+$', url))  # match top-level domains
    return domain_count

# Detect shortened URLs by checking if URL is from a shortening service (e.g., bit.ly, goo.gl)
def shortening_service(url):
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'is.gd']
    return 1 if any(service in url for service in shortening_services) else 0

# Count the occurrences of "https" in the URL
def count_https(url):
    return url.count('https')

# Count the occurrences of "http" in the URL
def count_http(url):
    return url.count('http')

# Count the number of "%" symbols in the URL
def count_per(url):
    return url.count('%')

# Count the number of "?" symbols in the URL
def count_ques(url):
    return url.count('?')

# Count the number of hyphens "-" in the URL
def count_hyphen(url):
    return url.count('-')

# Count the number of "=" symbols in the URL
def count_equal(url):
    return url.count('=')

# Calculate the length of the URL
def url_length(url):
    return len(url)

# Calculate the length of the hostname (domain name)
def hostname_length(url):
    netloc = urlparse(url).netloc
    return len(netloc)

# Check if the URL contains any suspicious words (commonly found in phishing URLs)
def suspicious_words(url):
    suspicious_keywords = ['login', 'bank', 'secure', 'account', 'verify', 'signin', 'payment']
    if any(word in url for word in suspicious_keywords):
        return 1
    return 0

# Count the number of digits in the URL
def digit_count(url):
    return sum(c.isdigit() for c in url)

# Count the number of letters (alphabetic characters) in the URL
def letter_count(url):
    return sum(c.isalpha() for c in url)

# Calculate the length of the first directory path (or file name) in the URL
def fd_length(url):
    path = urlparse(url).path
    if path:
        path_segments = path.split('/')
        return len(path_segments[0]) if path_segments else 0
    return 0

# Calculate the length of the Top Level Domain (TLD) in the URL
def tld_length(tld):
    return len(tld) if tld else 0

df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))
df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))
df['count.'] = df['url'].apply(lambda i: count_dot(i))
df['count-www'] = df['url'].apply(lambda i: count_www(i))
df['count@'] = df['url'].apply(lambda i: count_atrate(i))
df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))
df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))
df['count-https'] = df['url'].apply(lambda i: count_https(i))
df['count-http'] = df['url'].apply(lambda i: count_http(i))
df['count%'] = df['url'].apply(lambda i: count_per(i))
df['count?'] = df['url'].apply(lambda i: count_ques(i))
df['count-'] = df['url'].apply(lambda i: count_hyphen(i))
df['count='] = df['url'].apply(lambda i: count_equal(i))
df['url_length'] = df['url'].apply(lambda i: url_length(i))
df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))
df['count-digits'] = df['url'].apply(lambda i: digit_count(i))
df['count-letters'] = df['url'].apply(lambda i: letter_count(i))
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))
df['tld_length'] = df['url'].apply(lambda i: tld_length(get_tld(i, fail_silently=True)))

# Encode the target labels
lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])

# Feature columns
X = df[['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@', 'count_dir', 'count_embed_domian', 'short_url',
        'count-https', 'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length', 'hostname_length',
        'sus_url', 'fd_length', 'tld_length', 'count-digits', 'count-letters']]

# Target variable
y = df['type_code']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)

# Use LGBMClassifier with class_weight='balanced' to address class imbalance
lgb = LGBMClassifier(objective='multiclass', boosting_type='gbdt', n_jobs=5, silent=True, random_state=5, class_weight='balanced')

# Train the model
lgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lgb = lgb.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_lgb))
print(confusion_matrix(y_test, y_pred_lgb))

# Save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(lgb, f)

# Predict function for Flask
def predict(test_url):
    features_test = main(test_url)
    features_test = np.array(features_test).reshape((1, -1))
    pred = lgb.predict(features_test)
    if int(pred[0]) == 0:
        return "SAFE"
    elif int(pred[0]) == 1:
        return "DEFACEMENT"
    elif int(pred[0]) == 2:
        return "PHISHING"
    elif int(pred[0]) == 3:
        return "MALWARE"

def main(url):

    status = []

    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))

    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))

    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))

    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url,fail_silently=True)

    status.append(tld_length(tld))




    return status

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_url():
    url = request.form['url']
    prediction = predict(url)
    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
