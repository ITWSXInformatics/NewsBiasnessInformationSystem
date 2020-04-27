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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
import pickle
import pdb
import os
import json
from nltk.util import ngrams

# Constants
STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he", "on", "at"
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
    #print("Topic Vectors:")
    #print(topic_vecs)
    #print("Compare truth to prediction (truth, prediction)")
    #for i in range(0, len(labels)):
    #    print("({}, {})".format(labels[i], pred_labels[i]))
    #print("Truth Labels:")
    #print(labels)
    #print("Predicted Labels:")
    #print(pred_labels)
    print("accuracy on training data: ",
          f1_score(labels, pred_labels, average='weighted'))

    return


def train_model(documents, onehot_enc, labels):
    """

    :param documents:
    :param onehot_enc:
    :param labels:
    :return:
    """
    # Configuration variables, how many topics will we attempt to extract from
    # our documents.
    num_topics = 100

    # Start
    print('number of documents: ', len(documents))

    id2word = corpora.Dictionary(documents)
    print(len(id2word.token2id))
    id2word.filter_extremes(no_below=50, no_above=0.6)
    print(len(id2word.token2id))

    corpus = [id2word.doc2bow(doc) for doc in documents]
    onehot_labels = onehot_enc.transform(labels)

    print("starting LDA model")
    # plug into LDA model.
    # this can take a while with larger number of documents
    lda = LdaMulticore(num_topics=num_topics,
                   id2word=id2word,
                   corpus=corpus,
                   passes=5,
                       workers=2)
    print("topics:")
    for topic in lda.show_topics(num_topics=num_topics,
                                 num_words=20):  # print_topics():
        print(topic)
    lda.save("trained_ldamodel_nobelow15_noabove60")

    # print("getting topics for testing document")
    # topic_prediction = lda.get_document_topics(bow=corpus[0])

    # print(testing_text_raw)
    # print(topic_prediction)

    print("")
    print(
        "starting setup to train a classifier based on LDA topics for each document")

    topic_vecs = []

    # get topic matches and put them into vectors
    for i in range(len(documents)):
        top_topics = lda.get_document_topics(corpus[i],
                                             minimum_probability=0)

        #print(len(top_topics))
        topic_vec = [top_topics[i][1] for i in range(num_topics)]
        topic_vecs.append(topic_vec)

    # train basic logistic regression
    model = ExtraTreesClassifier(
        random_state=None,
        bootstrap=True,
        min_impurity_split=None,
        min_samples_leaf=1,
        criterion='gini',
        min_impurity_decrease=0.0,
        max_features=40,
        n_estimators=110,
        min_samples_split=5,
        n_jobs=2,
        class_weight='balanced',
        max_depth=None
    ).fit(topic_vecs, labels)
    with open('../saved_models/extratrees_lda1560.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, topic_vecs


def load_articles(articles_dir, mbfc_labels):
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
    # nlp = spacy.load('en_core_web_md')

    # use 10 days as a prototype
    dates = [f for f in Path(articles_dir).iterdir() if f.is_dir()]

    save_processed_dir = (PROJ_ROOT / 'data/articles/preprocessed').resolve()
    # choose whether to use preprocessed data or not for dates
    dates = [f for f in Path(save_processed_dir).iterdir() if f.is_dir()]
    # dates = [f for f in Path(articles_dir).iterdir() if f.is_dir()]


    # with nlp.disable_pipes("ner"):
    if True:
        for date in dates[:60]:  # parsing documents can take a while. training done through [:60]
            print(len(documents), len(labels))
            smalltest_dir = (articles_dir / date).resolve()

            publishers = [f for f in smalltest_dir.iterdir() if f.is_dir()]
            this_date = date.name

            (save_processed_dir / str(this_date)).mkdir(parents=True, exist_ok=True)
            thisdate_dir = (save_processed_dir / str(this_date)).resolve()
            for pub_articles in publishers:
                articles = [f for f in pub_articles.iterdir() if f.is_file()]
                if articles:
                    this_publisher = pub_articles.name

                    # print(this_publisher)
                    # skip if no label for publisher
                    if str(this_publisher) not in mbfc_labels.keys():
                        continue
                    else:
                        (thisdate_dir/str(this_publisher)).mkdir(parents=True, exist_ok=True)
                        thispublisher_dir = (thisdate_dir/str(this_publisher)).resolve()
                        for article in articles:
                            # comment out next 4 lines if not using preprocessed data
                            with article.open('r') as f:
                                text = json.load(f)
                            documents.append(text)
                            labels.append(mbfc_labels[this_publisher])
                            # # uncomment below here if not using preprocessed data
                            # this_article = article.name
                            #
                            # text = article.read_text()
                            #
                            # text = text.replace("\n", " ")
                            #
                            # # # preprocessing text
                            # lem_text =  [token.lemma_.lower() for token in nlp(text)
                            #             if not token.is_stop
                            #             and not token.is_punct
                            #             and not token.is_space
                            #             and not token.lemma_.lower() in STOPWORDS
                            #             and not token.pos_ == 'SYM'
                            #             and not token.pos_ == 'NUM']
                            #
                            # lem_text += ["_".join(w) for w in ngrams(lem_text, 2)]
                            #
                            # documents.append(lem_text)
                            #
                            # (save_processed_dir/str(this_date)/str(this_publisher)).mkdir(parents=True, exist_ok=True)
                            # try:
                            #     with (thispublisher_dir/str(this_article)).open("w+", encoding='utf-8') as proc_f:
                            #          json.dump(lem_text, proc_f)
                            # except FileNotFoundError:
                            #     print("{} wasn't found. ??????????".format((thispublisher_dir/str(this_article))))
                            #
                            # labels.append(mbfc_labels[this_publisher])
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
    raw_data = None

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

    inputs = parser.parse_args()

    if inputs.p:
        preprocess_data(inputs.article_dir,
                        os.path.join(inputs.data_dir, 'articles', 'articles'))

    ###########################################################################
    # This problem can be broken into the following steps:
    #     1) Load the desired documents and labels into memory
    #     2) Train our model to be able to classify a given document's
    #        biases
    #     3) Train a classifier to be able to extract the associated terms,
    #        phrases and topics for the biases present in the document
    ###########################################################################

    # Step 1) load our label data, form of a tuple of (lables, publisher_data)
    documents, labels, onehot_enc = load_data(inputs.data_dir, inputs.article_dir)

    # Step 2) Train our model
    model, topic_vector = train_model(documents, onehot_enc, labels)

    # Step 2 TEST STEP, confirm that our prediction is good
    predict_bias(model, topic_vector, labels)

    # Step 3) Extract the words


    # Do the tests
    #TEST_SET()

    return


if __name__ == "__main__":
    main()
