from pathlib import Path
from gensim.models.ldamodel import  LdaModel
# from gensim.models.ldamulticore import LdaMulticore
import gensim.corpora as corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import csv
import numpy as np
import spacy
import en_core_web_md

# expecting labels.csv in the ../data folder, and articles to be in ../data/articles/articles/ etc.
PROJ_ROOT = Path(__file__).parent.parent
DATA_DIR = (PROJ_ROOT / 'data').resolve()
ARTICLES_DIR = (DATA_DIR / 'articles' / 'articles').resolve()

# loading the spacy model is alternatively done by importing spacy then calling spacy.load('en
# change to en_core_web_lg or en_core_web_sm depending on what model you have downloaded
nlp = en_core_web_md.load()
# adding a few more stopwords / words we aren't interested in given the kinds of things that often show up in news articles
my_stopwords = {
    'say', 'not', 'like', 'go', "be", "have", "s"
}

# get data from labels csv into a couple dictionaries
labels = dict()
# publisher data keys should match the folder names for each publisher for a given day
publisher_data = dict()
with (DATA_DIR / 'labels.csv').resolve().open() as f:
    labelreader = csv.reader(f, delimiter=',')
    firstrow = True
    for row in labelreader:
        if firstrow:
            for ind in range(len(row)):
                labels[row[ind]] = ind-1
            firstrow = False
            continue
        publisher_data[row[0]] = row[1:]

# for lab in labels.keys():
#     print(lab)
print("===========")
# has labels like "left_bias", "right_center_bias", "questionable_source", "conspiracy_pseudoscience"
label_ind = labels['Media Bias / Fact Check, label']
mbfc_labels = dict()
for pub in publisher_data.keys():
    bias_label = publisher_data[pub][label_ind]
    if bias_label != '':
        # print(pub, ':', bias_label)
        mbfc_labels[pub] = bias_label
# print(publisher_data.keys())
print(set(mbfc_labels.values()))
print(len(set(mbfc_labels.values())))
onehot_enc = LabelBinarizer().fit(list(mbfc_labels.values()))

documents = []
labels = []

dates = [f for f in ARTICLES_DIR.iterdir() if f.is_dir()]
# use 10 days as a prototype

# store raw text / processed text to use later to look at result example
testing_text_raw = []

with nlp.disable_pipes("ner"):
    for date in dates[:2]: # parsing documents can take a while.
        smalltest_dir = (ARTICLES_DIR / date).resolve()

        publishers = [f for f in smalltest_dir.iterdir() if f.is_dir()]

        for pub_articles in publishers:
            articles = [f for f in pub_articles.iterdir() if f.is_file()]
            if articles:
                this_publisher = str(articles[0].parent.relative_to(smalltest_dir))
                # print(this_publisher)
                # skip if no label for publisher
                if str(this_publisher) not in mbfc_labels.keys():
                    continue
                else:
                    text = articles[0].read_text()

                    # save raw text of first document, just for looking at results later
                    if not testing_text_raw:
                        testing_text_raw = text

                    text = text.replace("\n", " ")

                    # preprocessing text
                    lem_text = [token.lemma_.lower() for token in nlp(text)
                                if not token.is_stop
                                and not token.is_punct
                                and not token.is_space
                                and not token.lemma_.lower() in my_stopwords
                                and not token.pos_ == 'SYM'
                                and not token.pos_ == 'NUM']
                    # print(lem_text)
                    documents.append(lem_text)
                    labels.append(mbfc_labels[this_publisher])

print('number of documents: ', len(documents))

id2word = corpora.Dictionary(documents)

corpus = [id2word.doc2bow(doc) for doc in documents]

onehot_labels = onehot_enc.transform(labels)

print("starting LDA model")
# plug into LDA model.
# this can take a while with larger number of documents
lda = LdaModel(num_topics=20,
                   id2word=id2word,
                   corpus=corpus,
                   passes=50,
                   eval_every=1)
print("topics:")
for topic in lda.show_topics(num_topics=20, num_words=20):#print_topics():
    print(topic)

#print("getting topics for testing document")
#topic_prediction = lda.get_document_topics(bow=corpus[0])

#print(testing_text_raw)
#print(topic_prediction)

print("")
print("starting setup to train a classifier based on LDA topics for each document")

topic_vecs = []

# get topic matches and put them into vectors
for i in range(len(documents)):
    top_topics = lda.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    topic_vecs.append(topic_vec)


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score

# train basic logistic regression
sgd_model = LogisticRegression(class_weight='balanced').fit(topic_vecs, labels)

pred_labels = sgd_model.predict(topic_vecs)

# get accuracy from the training data, just to look at whether this even seems feasible...
# 0.3 f1 score on the training, using 12123 documents. not great results for now.
print("accuracy on training data: ", f1_score(labels, pred_labels, average='weighted'))