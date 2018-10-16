import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
'''
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print("\n")
'''
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])

print(messages.head())
print(messages.describe())
print(messages.groupby('label').describe())


messages['length'] = messages['message'].apply(len)
sns.distplot(messages['length'],kde=False,bins=100)
plt.show()

print(messages['length'].describe())

messages.hist(column='length',by='label',bins=60)
plt.show()

def text_process(mess):
    """
    1. Remove Punc
    2. Remove Stop words
    3. Return list of clean text words
    """

    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [ word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''
bagofwords_tranformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bagofwords_tranformer.vocabulary_))
mess4 = messages['message'][3]
bow4 = bagofwords_tranformer.transform([mess4])
print(bow4)

messages_bow = bagofwords_tranformer.transform(messages['message'])

print("Shape of Sparse Matrix: ", messages_bow.shape)
print(messages_bow.nnz)

tfidf_transformer = TfidTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])

all_pred = spam_detect_model.predict(messages_tfidf)
'''
msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidif',TfidfTransformer()),
    ('classifier',MultinomialNB())
    ])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(label_test,predictions))
