from gensim.models.ldamodel import LdaModel
import argparse
import spacy
import numpy as np
import pickle
from nltk.util import ngrams
from pathlib import Path
import BiasDetector
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
    with open(inputs.classifier_model, 'rb') as f:
        logreg = pickle.load(f)
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

    corpus = [id2word.doc2bow(doc) for doc in documents]
    topic_vecs = []
    for doc_as_corpus in corpus:
        top_topics = lda.get_document_topics(doc_as_corpus,
                                             minimum_probability=0)
        topic_vec = [top_topics[i][1] for i in range(lda.num_topics)]
        topic_vecs.append(topic_vec)#.reshape(1, -1))
        # for word in doc_as_corpus:
        #     print(id2word[word[0]], lda.get_term_topics(word[0]))

    # for i in range(len(topic_vecs)):
    #     #print(raw_texts[i])
    #     print(topic_vecs[i])
    #     print("prediction: {}".format(logreg.predict(topic_vecs[i])))

    print("---topics")
    for topic in lda.show_topics(num_topics=lda.num_topics,
                                 num_words=15):  # print_topics():
        print(topic)
    with open('../saved_models/trained_ExtraTree_model_boostrap_minsamplesleaf5_entropy.pkl', 'rb') as f:
        model = pickle.load(f)

    pred_labels = model.predict(topic_vecs)
    BiasDetector.predict_bias(model, topic_vecs, labels)

    # print("0")
    # new_model = RandomForestClassifier(max_depth=10, max_features=None, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_RFC_model_maxdepth10_maxfeaturesnone.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("1")
    # new_model = RandomForestClassifier(max_depth=5, max_features='auto', n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_RFC_model_maxdepth5_maxfeaturesauto.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("2")
    # new_model = RandomForestClassifier(max_depth=10, max_features='auto', n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_RFC_model_maxdepth10_maxfeaturesauto.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("3")
    # new_model = RandomForestClassifier(max_depth=5, max_features=10, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_RFC_model_maxdepth5_maxfeatures10.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("4")
    # new_model = ExtraTreesClassifier(max_depth=5, min_samples_leaf=10, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_maxdepth5_minsamplesleaf10.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("5")
    # new_model = ExtraTreesClassifier(max_depth=10, min_samples_leaf=10, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_maxdepth10_minsamplesleaf10.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("6")
    # new_model = ExtraTreesClassifier(max_depth=10, min_samples_leaf=5, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_maxdepth10_minsamplesleaf5.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    # print("7")
    # new_model = ExtraTreesClassifier(max_features='auto', bootstrap=True, min_samples_leaf=5, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_bootstrap_minsamplesleaf5.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # print("8")
    # new_model = ExtraTreesClassifier(max_features='auto', bootstrap=True, min_samples_leaf=5, n_estimators=100, class_weight='balanced', criterion='entropy').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_boostrap_minsamplesleaf5_entropy.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)
    #
    # print("9")
    # new_model = ExtraTreesClassifier(max_features='auto', bootstrap=True, n_estimators=100, class_weight='balanced').fit(topic_vecs, labels) #trained on [:60]
    #
    # with open('../saved_models/trained_ExtraTree_model_bootstrap.pkl', 'wb') as f:
    #     pickle.dump(new_model, f)
    # pred_labels = new_model.predict(topic_vecs)
    #
    # BiasDetector.predict_bias(new_model, topic_vecs, labels)


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