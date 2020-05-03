from gensim.models.ldamodel import LdaModel
import argparse
import spacy
import numpy as np
import pickle
import os

STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he"
}


def loadModules(models_dir="models"):
    """
    Load our LDA and

    :return:
    """
    lda_path = os.path.join(models_dir, "LDA_Model.model")
    classifier_path = os.path.join(models_dir, "Classifer.pkl")


    print("Loading LDA model")
    lda = LdaModel.load(lda_path)
    id2word = lda.id2word
    print("Finished loading LDA model")

    print("Loading Classifier")
    try:
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
            print("Finished loading classifier!")
    except:
        classifer = None
        print("Failed to load classifier!")

    print("Loading spacy")
    nlp = spacy.load('en_core_web_md')
    print("Finished loading spacy")

    return lda, classifer, nlp


def predict(article_data):
    """
    Predict the class and topics of the given article data.

    :param article_data:
    :return:
    """

    lda, classifier, nlp = loadModules()
    id2word = lda.id2word

    documents = []
    raw_texts = []
    with nlp.disable_pipes("ner"):
        text_raw = article_data['text']
        raw_texts.append(text_raw)

        text = text_raw.replace("\n", " ")

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

    corpus = [id2word.doc2bow(doc) for doc in documents]
    topic_vecs = []
    for doc_as_corpus in corpus:
        top_topics = lda.get_document_topics(doc_as_corpus,
                                             minimum_probability=0)
        topic_vec = [top_topics[i][1] for i in range(lda.num_topics)]
        topic_vecs.append(np.array(topic_vec).reshape(1, -1))

    for i in range(len(topic_vecs)):
        print(raw_texts[i])
        print(topic_vecs[i])
        print("prediction: {}".format(classifier.predict(topic_vecs[i])))

    return


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("lda_model",
                        type=str,
                        help=("The LDA model file to load"))

    parser.add_argument("classifier_model",
                        type=str,
                        help=("The classifier model file to load"))


    inputs = parser.parse_args()

    print("loading LDA model")
    lda = LdaModel.load(inputs.lda_model)
    id2word = lda.id2word
    print("finished loading lda model")

    print("loading logistic regression model")
    with open(inputs.classifier_model, 'rb') as f:
        logreg = pickle.load(f)
    print("finished logistic regression model")

    print("loading spacy")
    nlp = spacy.load('en_core_web_md')
    print("finished loading spacy")

    # load up a file, process, then predict.
    # modify this file to something that exists on your machine
    dummy_file = 'E:/Programming/PyCharmProjects/NewsBiasnessInformationSystem/data/articles/articles/2018-02-01/Addicting Info/Addicting Info--2018-02-01--Donald Trump Jr Likes Fox News Tweet About Spread Of Russian Propaganda'

    documents = []
    raw_texts = []
    with nlp.disable_pipes("ner"):
        with open(dummy_file, 'r') as f:
            text_raw = f.read()
            raw_texts.append(text_raw)

            text = text_raw.replace("\n", " ")

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

    corpus = [id2word.doc2bow(doc) for doc in documents]
    topic_vecs = []
    for doc_as_corpus in corpus:
        top_topics = lda.get_document_topics(doc_as_corpus,
                                             minimum_probability=0)
        topic_vec = [top_topics[i][1] for i in range(lda.num_topics)]
        topic_vecs.append(np.array(topic_vec).reshape(1, -1))

    for i in range(len(topic_vecs)):
        print(raw_texts[i])
        print(topic_vecs[i])
        print("prediction: {}".format(logreg.predict(topic_vecs[i])))


if __name__ == "__main__":
    main()
