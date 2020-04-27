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

DUMMY_TEXT = """
Russia has appointed the US actor Steven Seagal as a special envoy to improve ties with the United States.

Seagal was granted Russian citizenship in 2016 and has praised President Putin as a great world leader.

Born in the US, the martial arts star gained international fame for roles in the 1980s and '90s like Under Siege.

He is also one of the Hollywood stars accused by several women of sexual misconduct in the wake of the #MeToo campaign, which he has denied.

The Russian foreign ministry made the announcement on its official Facebook page, saying the unpaid position was similar to that of a United Nations' goodwill ambassador and Seagal would promote US-Russia relations "in the humanitarian sphere".

The Flight of Fury star, still popular with Russian audiences, has recently defended the Russian government over claims that it meddled in 2016 US elections.

The 66-year-old has called President Putin "one of the great living world leaders", and when Seagal was granted Russian citizenship, said he hoped it would be a symbol of how relations between Moscow and Washington were starting to improve.

Seagal was also granted Serbian citizenship in 2016, following several visits to the Balkan country.
"""

STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he"
}

def get_json_prediction_output(nlp, lda_model, classifier_model, input_raw_text):

    text = input_raw_text.replace("\n", " ")

    # # preprocessing text
    lem_text =  [token.lemma_.lower() for token in nlp(text)
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and not token.lemma_.lower() in STOPWORDS
                and not token.pos_ == 'SYM'
                and not token.pos_ == 'NUM']

    lem_text += ["_".join(w) for w in ngrams(lem_text, 2)]
    documents = [lem_text]

    corpus = [lda_model.id2word.doc2bow(doc) for doc in documents]
    topic_vecs = []
    output_overall_topics = []
    output_word_topics = []
    relevant_topic_details = []
    for doc_as_corpus in corpus:
        top_topics = lda_model.get_document_topics(doc_as_corpus,
                                             minimum_probability=0)
        topic_vec = [top_topics[i][1] for i in range(lda_model.num_topics)]
        topic_vecs.append(topic_vec)

        relevant_topics = lda_model.get_document_topics(doc_as_corpus)
        output_overall_topics.append([topic[0] for topic in relevant_topics])
        for topic in relevant_topics:
            top_term_ids = lda_model.get_topic_terms(topic[0])
            top_terms = [lda_model.id2word[tup[0]] for tup in top_term_ids]
            relevant_topic_details.append((topic[0], top_terms))
            print("==>",(topic[0], top_terms))

        for word_tuple in doc_as_corpus:
            word_topics = lda_model.get_term_topics(word_tuple[0])
            if word_topics:
                output_word_topics.append((lda_model.id2word[word_tuple[0]], [wt[0] for wt in word_topics]))

    output_pred_label = classifier_model.predict(topic_vecs)[0]

    output_dict = {'pred_label': output_pred_label,
                   'overall_doc_topics': output_overall_topics,
                   'relevant_topic_terms': relevant_topic_details,
                   'per_word_topics': output_word_topics}
    print(output_dict)

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
    nlp = spacy.load('en_core_web_md')
    print("finished loading spacy")

    mbfc_labels, onehot_enc = BiasDetector.load_labels('../data/')

    # load up a file, process, then predict.
    # modify this file to something that exists on your machine
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
        for date in dates[60:61]:
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
        topic_vecs.append(topic_vec)

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
    # joblib.dump(GS.best_estimator_, 'gridsearch_extratrees_bestmodel_paramsv2_filterextremes.pkl')
    # print(GS.cv_results_)
    print("loading saved gridsearch model")
    GS = joblib.load(inputs.classifier_model) #_paramsv2
    print(GS.get_params())
    pred_labels = GS.predict(topic_vecs)
    BiasDetector.predict_bias(GS, topic_vecs, labels)

    for l in set(mbfc_labels.values()):
        print(l, ':', labels.count(l)/len(labels))

    print(get_json_prediction_output(nlp, lda, GS, DUMMY_TEXT))

if __name__ == "__main__":
    main()

