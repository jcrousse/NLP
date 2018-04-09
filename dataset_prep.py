from collections import Counter
import numpy as np
import pandas as pd
import numba


reviews_file = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],reviews_file.readlines()))
reviews_file.close()

labels_file = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),labels_file.readlines()))
labels_file.close()

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1


pos_neg_ratios = Counter()

# Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = np.log(positive_counts[term] / float(negative_counts[term]+1))
        pos_neg_ratios[term] = pos_neg_ratio

vocab = set(total_counts.keys())
vocab_size = len(vocab)

word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i


def review_to_input(review):

    layer_0 = np.zeros((1, vocab_size))

    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1

    return layer_0


@numba.vectorize
def label_to_target(label):
    if(str(label == 'POSITIVE')):
        return 1
    else:
        return 0

series_label = pd.Series(labels)
df_label= pd.get_dummies(series_label)
binary_target = df_label["POSITIVE"]

#todo next time: Batch data then Keras model
def get_batches(x, y, batch_size=1000):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for idx in range(0, len(x), batch_size):
        yield x[idx:idx + batch_size], y[idx:idx + batch_size]

for idx, (train, target) in enumerate(get_batches(reviews, labels)):
    print("running batch ", idx)
    print("transforming reviews to numbers ...")

layer_inputs = []
for review in reviews[1:10]:
    layer_inputs.append(review_to_input(review).flatten())

df_out = pd.DataFrame(layer_inputs)
print(df_out.head())
df_out.to_csv("csv_out_test.csv")

#X = np.hstack(layer_inputs)
#np.savetxt("BagOfWords_X_2.csv", X, delimiter=",")

#print("creating dataframe ...")
#pd.DataFrame(layer_inputs)

#X = np.hstack(layer_inputs)
#np.savetxt("BagOfWords_X.csv", X, delimiter=",")