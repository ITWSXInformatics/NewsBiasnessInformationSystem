from gensim.models.ldamodel import LdaModel
import argparse
import spacy
import numpy as np
import pickle
from nltk.util import ngrams
from pathlib import Path
import BiasDetector
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import json

STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he"
}

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
    # with open(inputs.classifier_model, 'rb') as f:
    #     logreg = pickle.load(f)
    print("finished logistic regression model")

    print("loading spacy")
    # nlp = spacy.load('en_core_web_md')
    print("finished loading spacy")

    mbfc_labels, onehot_enc = BiasDetector.load_labels('../data/')

    # load up a file, process, then predict.
    # modify this file to something that exists on your machine
    dummy_file = 'E:/Programming/PyCharmProjects/NewsBiasnessInformationSystem/data/articles/articles/2018-02-01/Addicting Info/Addicting Info--2018-02-01--Donald Trump Jr Likes Fox News Tweet About Spread Of Russian Propaganda'
    PROJ_ROOT = Path(__file__).parent.parent
    articles_dir = (PROJ_ROOT / 'data/articles/articles').resolve()
    save_processed_dir = (PROJ_ROOT / 'data/articles/preprocessed').resolve()
    # choose whether to use preprocessed data or not for dates
    dates = [f for f in Path(save_processed_dir).iterdir() if f.is_dir()]
    # dates = [f for f in Path(articles_dir).iterdir() if f.is_dir()]

    documents = []
    labels = []
    raw_texts = []
    # with nlp.disable_pipes("ner"):
    if True:
        for date in dates[60:70]:#dates[:60]:  # parsing documents can take a while. training done through [:60]
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
                            # # comment out next 4 lines if not using preprocessed data
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

    corpus = [id2word.doc2bow(doc) for doc in documents]
    topic_vecs = []
    has_topic_count = 0
    for doc_as_corpus in corpus:
        top_topics = lda.get_document_topics(doc_as_corpus,
                                             minimum_probability=0)
        topic_vec = [top_topics[i][1] for i in range(lda.num_topics)]
        topic_vecs.append(topic_vec)#.reshape(1, -1))
        # print("----------")
        # print("document topics: ", lda.get_document_topics(doc_as_corpus))
        # #print(doc_as_corpus)
        # for word in doc_as_corpus:
        #     topics = lda.get_term_topics(word[0])
        #     if topics:
        #         has_topic_count += 1
        #         print(id2word[word[0]], topics)

    # for i in range(len(topic_vecs)):
    #     #print(raw_texts[i])
    #     print(topic_vecs[i])
    #     print("prediction: {}".format(logreg.predict(topic_vecs[i])))

    print("---topics")
    for topic in lda.show_topics(num_topics=lda.num_topics,
                                 num_words=15):  # print_topics():
        print(topic)
    print("starting gridsearch")
    # params = {'n_estimators': [110],
    #      'max_depth':[None, 10],
    #           'criterion':['gini', 'entropy'],
    #           'min_samples_leaf': [1, 2, 5],
    #           'min_samples_split': [2,5,10],
    #           'max_features':['auto', 10, 20, 40],
    #           'bootstrap':[True],
    #           'class_weight': ['balanced']}
    # ETC = ExtraTreesClassifier()
    # GS = GridSearchCV(ETC, params, n_jobs=2)
    # GS.fit(topic_vecs, labels)
    #
    # joblib.dump(GS.best_estimator_, '../saved_models/gridsearch_extratrees_bestmodel_paramsv2_filterextremes.pkl')
    # print(GS.cv_results_)
    print("loading saved gridsearch model")
    GS = joblib.load('../saved_models/gridsearch_extratrees_bestmodel_paramsv2_filterextremes.pkl') #_paramsv2
    print(GS.get_params())
    pred_labels = GS.predict(topic_vecs)
    BiasDetector.predict_bias(GS, topic_vecs, labels)

    # print(pred_labels, labels)

    for l in set(mbfc_labels.values()):
        print(l, ':', labels.count(l)/len(labels))


if __name__ == "__main__":
    main()


"""

0
accuracy on training data:  0.4962224023275393
1
accuracy on training data:  0.2936234498922573
2
accuracy on training data:  0.5413597100467187
3
accuracy on training data:  0.2891795972620253
4
accuracy on training data:  0.2452928005483607
5
accuracy on training data:  0.30321583839316874
6
accuracy on training data:  0.33232247964733097
7
accuracy on training data:  0.32341255562835003
8
accuracy on training data:  0.3598775280932731
"""