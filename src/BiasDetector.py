from pathlib import Path
from gensim.models.ldamodel import  LdaModel
from gensim.models.ldamulticore import LdaMulticore
import gensim.corpora as corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import csv
import numpy as np
import spacy
#import en_core_web_md
import argparse
import json
#import sqlite3
from sklearn.linear_model import (
    SGDClassifier, LogisticRegression
)
from sklearn.svm import (
    LinearSVC
)

from sklearn.metrics import f1_score
import pickle
import random
import os
import json
import pdb

# Constants
STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he"
}


def preprocess_data(article_directory, dest_root_dir):
    """

    :param article_directory:
    :return:
    """
    date_to_article = {}
    dir = os.fsencode(article_directory)

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        publisher = filename.split('.')[0]
        with open(os.path.join(article_directory, filename)) as f:
            all_articles = json.load(f)
            for article in all_articles:
                date = article['date']
                article_id = article['title'][:200] if len(article['title']) > 200 else article['title']
                article_id = article_id.replace('/', '_')
                date_to_article[date] = article

                # Check if dirs exist
                date_dir = os.path.join(dest_root_dir, date)
                publisher_dir = os.path.join(dest_root_dir, date, publisher)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir, mode=0o777)

                if not os.path.exists(publisher_dir):
                    os.makedirs(publisher_dir, mode=0o777)

                # Write article content to text file
                try:
                    article_file = os.path.join(publisher_dir, article_id+'.txt')
                    if not os.path.isfile(article_file):
                        with open(article_file, 'w') as out:
                            out.write(article['content'])
                except Exception as e:
                    print("ERROR:", e)

    return


def predict_bias(model, topic_vecs, labels):
    """

    :param model:
    :param topic_vecs:
    :param labels:
    :return:
    """

    pred_labels = model.predict(topic_vecs)

    # get accuracy from the training data, just to look at whether this even seems feasible...
    # 0.3 f1 score on the training, using 12123 documents. not great results for now.
    print("Topic Vectors:")
    print(topic_vecs)
    print("Compare truth to prediction (truth, prediction)")
    # for i in range(0, len(labels)):
    #     print("({}, {})".format(labels[i], pred_labels[i]))
    #print("Truth Labels:")
    #print(labels)
    #print("Predicted Labels:")
    #print(pred_labels)
    print("accuracy on training data: ",
          f1_score(labels, pred_labels, average='weighted'))

    return


def train_model(documents, onehot_enc, labels, hyperparams=None):
    """

    :param documents:
    :param onehot_enc:
    :param labels:
    :return:
    """
    # Configuration variables, how many topics will we attempt to extract from
    # our documents.
    if hyperparams is not None:
        num_topics = hyperparams['num_topics']
        num_words = hyperparams['num_words']
        filter_extremes = hyperparams['filter_extremes']
        classifier = hyperparams['classifier']
    else:
        num_topics = 400
        num_words = 20
        filter_extremes = False
        classifier = 'logreg'

    # Start
    print('number of documents: ', len(documents))

    id2word = corpora.Dictionary(documents)

    if filter_extremes:
        # dynamically filter out unnecessary words (similar to stop words)
        id2word.filter_extremes(no_below=5, no_above=0.4)#, keep_n=10000)

    corpus = [id2word.doc2bow(doc) for doc in documents]
    onehot_labels = onehot_enc.transform(labels)

    print("starting LDA model")
    # plug into LDA model.
    # this can take a while with larger number of documents
    lda = LdaMulticore(num_topics=num_topics,
                   id2word=id2word,
                   corpus=corpus,
                   passes=50,
                   eval_every=1,
                   workers=3)
    print("topics:")
    for topic in lda.show_topics(num_topics=num_topics,
                                 num_words=num_words):  # print_topics():
        print(topic)

    print("Save LDA model.")
    filename = "trained_ldamodel_"+str(num_topics)+'_'+str(num_words)+"_"+str(filter_extremes)+".model"
    traits = "config_" + str(num_topics)+'_'+str(num_words)+"_"+str(filter_extremes)
    path = os.path.join('models', traits)
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)
    lda.save(os.path.join('models', traits, filename))
    print("Finish save!")

    # print("getting topics for testing document")
    # topic_prediction = lda.get_document_topics(bow=corpus[0])

    # print(testing_text_raw)
    # print(topic_prediction)

    # Classification step
    model, topic_vecs = train_classifier(lda, documents, corpus, labels, hyperparams)

    return model, lda, topic_vecs, corpus


def train_classifier(lda, documents, corpus, labels, hyperparams):
    """

    :return:
    """

    print("")
    print(
        "starting setup to train a classifier based on LDA topics for each document")

    num_topics = hyperparams['num_topics']
    num_words = hyperparams['num_words']
    filter_extremes = hyperparams['filter_extremes']
    classifier = hyperparams['classifier']

    topic_vecs = []

    # get topic matches and put them into vectors
    for i in range(len(documents)):
        top_topics = lda.get_document_topics(corpus[i],
                                             minimum_probability=0)

        # print(len(top_topics))
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        topic_vecs.append(topic_vec)

    if classifier == 'logreg':
        # train basic logistic regression
        model = LogisticRegression(class_weight='balanced',
                                   n_jobs=3).fit(topic_vecs, labels)
        with open('trained_logreg_model_' + str(num_topics) + '_' + str(
                num_words) + "_" + str(filter_extremes) + '.pkl', 'wb') as f:
            pickle.dump(model, f)
    if classifier == 'svm':
        model = LinearSVC(class_weight='balanced').fit(topic_vecs, labels)
        with open('trained_linsvc_model_' + str(num_topics) + '_' + str(
                num_words) + "_" + str(filter_extremes) + '.pkl', 'wb') as f:
            pickle.dump(model, f)

    return model, topic_vecs


def get_document_topics(documents, corpus, lda_model, num_topics):
    """
    Given a set of documents, determine the topic content vector
    (vector of the document topic probabilities)

    :param documents:
    :param corpus:
    :param lda_model:
    :return:
    """
    topic_vecs = []
    # get topic matches and put them into vectors
    for i in range(len(documents)):
        top_topics = lda_model.get_document_topics(corpus[i],
                                                   minimum_probability=0)

        # print(len(top_topics))
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        topic_vecs.append(topic_vec)

    return topic_vecs


def load_articles(articles_dir, mbfc_labels, num_articles=100):
    """

    :param articles_dir:
    :return:
    """
    # initialize return values.
    documents = []
    labels = []
    # store raw text / processed text to use later to look at result example
    testing_text_raw = []
    PROJ_ROOT = Path(__file__).parent.parent
    print(articles_dir)
    articles_dir = (PROJ_ROOT / articles_dir).resolve()
    print("--")
    print(articles_dir)

    # load our spacy model
    nlp = spacy.load('en_core_web_md')

    # use 10 days as a prototype
    dates = [f for f in Path(articles_dir).iterdir() if f.is_dir()]
    selected_dates = random.sample(range(364), num_articles)
    print(selected_dates)

    with nlp.disable_pipes("ner"):
        for i in selected_dates:  # parsing documents can take a while.
            date = dates[i]
            smalltest_dir = (articles_dir / date).resolve()

            publishers = [f for f in smalltest_dir.iterdir() if f.is_dir()]

            for pub_articles in publishers:
                articles = [f for f in pub_articles.iterdir() if f.is_file()]
                if articles:
                    for i in range(min(len(articles), 20)):
                        article = articles[i]
                        this_publisher = str(
                            article.parent.relative_to(smalltest_dir))
                        # print(this_publisher)
                        # skip if no label for publisher
                        if str(this_publisher) not in mbfc_labels.keys():
                            continue
                        else:
                            text = article.read_text()

                            # save raw text of first document, just for looking at results later
                            if not testing_text_raw:
                                testing_text_raw = text

                            text = text.replace("\n", " ")

                            # preprocessing text
                            lem_text = [token.lemma_.lower() for token in nlp(text)
                                        if not token.is_stop
                                        and not token.is_punct
                                        and not token.is_space
                                        and not token.lemma_.lower() in STOPWORDS
                                        and not token.pos_ == 'SYM'
                                        and not token.pos_ == 'NUM']
                            # print(lem_text)
                            documents.append(lem_text)
                            labels.append(mbfc_labels[this_publisher])
    return documents, labels


def load_labels(path):
    """

    :param path:
    :return:
    """

    # get data from labels csv into a couple dictionaries
    labels = dict()
    # publisher data keys should match the folder names for each publisher for a given day
    publisher_data = dict()
    #with (path / 'labels.csv').resolve().open() as f:
    with open(os.path.join(path, 'labels.csv')) as f:
        labelreader = csv.reader(f, delimiter=',')
        firstrow = True
        for row in labelreader:
            if firstrow:
                for ind in range(len(row)):
                    labels[row[ind]] = ind - 1
                firstrow = False
                continue
            publisher_data[row[0]] = row[1:]


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

    return mbfc_labels, onehot_enc


def load_data(data_dir, article_dir):
    """
    This method will load in our biased news dataset, either as a json blob
    or as a sqlite database.

    :param path:
    :param mode:
    :return:
    """

    # load our news source labels
    mbfc_labels, onehot_enc = load_labels(data_dir)

    # Load the articles into memory
    documents, labels = load_articles(article_dir, mbfc_labels)

    return documents, labels, onehot_enc


def main():
    # Using the argument parser library to handle command line inputs
    # Note: this will also handle validating the input types
    program_desc = ("This program will train an NLP model that can classify "
                    "a document as containing some bias and then identify "
                    "corresponding terms that indicate said bias in "
                    "the article.")
    parser = argparse.ArgumentParser(description=program_desc)

    parser.add_argument("data_dir",
                        type=str,
                        help=("A text file containing our bias dataset."))

    parser.add_argument("article_dir",
                        type=str,
                        help=("Directory the articles live in."))

    parser.add_argument("-p",
                        action='store_true',
                        help=("Preprocess the dataverse files into a "
                              "format this program can use. Warning! This "
                              "will take up ~4GB of space on this disk."))

    parser.add_argument("--load",
                        #action='store_true',
                        nargs='?',
                        const=1,
                        type=str,
                        help=("Attempt to load pre-trained models in the "
                              "current directory."))

    parser.add_argument("--train_classifier",
                        # action='store_true',
                        nargs='?',
                        const=1,
                        type=str,
                        help=("Attempt to load pre-trained lda in the "
                              "current directory."))

    inputs = parser.parse_args()

    # TEMP, DO HYPERPARAMETER TEST
    #TEST_HYPERPARAMS(inputs)
    #return

    if inputs.p:
        preprocess_data(inputs.article_dir,
                        os.path.join(inputs.data_dir, 'articles', 'articles'))

    if inputs.train_classifier:
        files = [f for f in os.listdir(inputs.train_classifier) if
                 os.path.isfile(os.path.join(inputs.train_classifier, f))]

        lda_model_path = None
        for file in files:
            if "ldamodel" in file and ".exp" not in file and "id2" not in file and "state" not in file:
                lda_model_path = os.path.join(inputs.train_classifier, file)

        print("loading LDA model")
        # lda_model = LdaModel.load('trained_ldamodel.model')
        lda_model = LdaModel.load(lda_model_path)
        id2word = lda_model.id2word
        print("finished loading lda model")

        print("loading spacy")
        nlp = spacy.load('en_core_web_md')
        print("finished loading spacy")

        documents, labels, onehot_enc = load_data(inputs.data_dir, inputs.article_dir)

        corpus = [id2word.doc2bow(doc) for doc in documents]
        num_topics = 1000
        hyperparams = {'num_topics': num_topics, 'num_words': 20,
                       'filter_extremes': True, 'classifier': 'svm'}

        model, topic_vectors = train_classifier(lda_model, documents, corpus,
                                                labels, hyperparams)
        predict_bias(model, topic_vectors, labels)

    ###########################################################################
    # This problem can be broken into the following steps:
    #     1) Load the desired documents and labels into memory
    #     2) Train our model to be able to classify a given document's
    #        biases
    #     3) Train a classifier to be able to extract the associated terms,
    #        phrases and topics for the biases present in the document
    ###########################################################################
    elif inputs.load is not None:

        files = [f for f in os.listdir(inputs.load) if os.path.isfile(os.path.join(inputs.load, f))]

        lda_model_path = None
        logreg_model_path = None
        for file in files:
            if "ldamodel" in file and ".exp" not in file and "id2" not in file and "state" not in file:
                lda_model_path = os.path.join(inputs.load, file)
            elif "linsvc" in file:
                logreg_model_path = os.path.join(inputs.load, file)

        print("loading LDA model")
        #lda_model = LdaModel.load('trained_ldamodel.model')
        lda_model = LdaModel.load(lda_model_path)
        id2word = lda_model.id2word
        print("finished loading lda model")

        print("loading logistic regression model")
        with open(logreg_model_path, 'rb') as f:
            logreg_model = pickle.load(f)
        print("finished logistic regression model")

        print("loading spacy")
        nlp = spacy.load('en_core_web_md')
        print("finished loading spacy")

        load_data_params = [inputs.data_dir, inputs.article_dir]
        TEST_SET(logreg_model, lda_model, id2word, load_data_params)

    else:
        hyperparams = {'num_topics': 1000, 'num_words': 30,
                       'filter_extremes': True, 'classifier': 'svm'}
        # Step 1) load our label data, form of a tuple of (lables, publisher_data)
        documents, labels, onehot_enc = load_data(inputs.data_dir, inputs.article_dir)

        # Step 2) Train our model
        logreg_model, lda_model, topic_vector, corpus = train_model(
            documents, onehot_enc, labels, hyperparams)

        # Step 2 TEST STEP, confirm that our prediction is good
        predict_bias(logreg_model, topic_vector, labels)

        # Step 3) Extract the words


    # Do the tests
    #load_data_params = [inputs.data_dir, inputs.article_dir]
    #TEST_SET(logreg_model, lda_model, id2word, load_data_params)

    return


def TEST_HYPERPARAMS(inputs):
    """
    Generate a set of trained models with varying hyperparameters
    :return:
    """
    # load test documents to see performance on unknown documents
    #mbfc_labels, onehot_enc = load_labels(inputs.data_dir)
    #test_docs, test_labels = load_articles(inputs.article_dir, mbfc_labels, slice(50, 70))

    #hyperparams = {
    #    'num_topics': 10,
    #    'num_words': 20,
    #    'filter_extremes': True
    #}
    cases = [
        #{'num_topics': 10, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 10, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 15, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 15, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 20, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 20, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 30, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 30, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 70, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 70, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 120, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 120, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 160, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 160, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 200, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 200, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 300, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 300, 'num_words': 20, 'filter_extremes': False},
        {'num_topics': 400, 'num_words': 20, 'filter_extremes': True, 'classifier': 'svm'},
        #{'num_topics': 400, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 600, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 600, 'num_words': 20, 'filter_extremes': False},

        #{'num_topics': 20, 'num_words': 10, 'filter_extremes': True},
        ##{'num_topics': 20, 'num_words': 10, 'filter_extremes': False},
        #{'num_topics': 20, 'num_words': 30, 'filter_extremes': True},
        ##{'num_topics': 20, 'num_words': 30, 'filter_extremes': False},
        #{'num_topics': 20, 'num_words': 50, 'filter_extremes': True},
        #{'num_topics': 20, 'num_words': 70, 'filter_extremes': True},
        #{'num_topics': 1000, 'num_words': 20, 'filter_extremes': True},
        ## {'num_topics': 1000, 'num_words': 20, 'filter_extremes': False},
        #{'num_topics': 2000, 'num_words': 20, 'filter_extremes': True},
        #{'num_topics': 4000, 'num_words': 20, 'filter_extremes': True},

    ]

    # Step 1) load our label data, form of a tuple of (lables, publisher_data)
    documents, labels, onehot_enc = load_data(inputs.data_dir,
                                              inputs.article_dir)

    for hyperparams in cases:
        print("Case:", hyperparams)


        # Step 2) Train our model
        logreg_model, lda_model, topic_vector, corpus = train_model(
            documents, onehot_enc, labels, hyperparams=hyperparams)
        print("Finish case")



    return


def TEST_SET(logreg_model, lda_model, id2word, load_documents_params):
    """

    :return:
    """
    print("Begin Tests!")

    num_topics = lda_model.num_topics
    mbfc_labels, onehot_enc = load_labels(load_documents_params[0])
    load_doc_params = (load_documents_params[1], mbfc_labels, 100)
    documents, labels = load_articles(*load_doc_params)
    corpus = [id2word.doc2bow(doc) for doc in documents]
    topic_vector = get_document_topics(documents, corpus, lda_model, num_topics)

    predict_bias(logreg_model, topic_vector, labels)

    return


if __name__ == "__main__":
    main()
