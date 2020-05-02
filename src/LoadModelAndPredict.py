from gensim.models.ldamodel import LdaModel
import argparse
import spacy
import numpy as np
import pickle
from nltk.util import ngrams


STOPWORDS = {
    'say', 'not', 'like', 'go', "be", "have", "s", #original
    "and", "when", "where", "who", "let", "look", "time", "use", "him", "her",
    "she", "he"
}

def preprocess_text(nlp, input_raw_text):
    text = input_raw_text.replace("\n", " ")

    # # preprocessing text
    lem_text = [token.lemma_.lower() for token in nlp(text)
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and not token.lemma_.lower() in STOPWORDS
                and not token.pos_ == 'SYM'
                and not token.pos_ == 'NUM']

    lem_text += ["_".join(w) for w in ngrams(lem_text, 2)]

    return lem_text

def get_json_prediction_output(nlp, lda_model, classifier_model, input_raw_text):

    pp_text = preprocess_text(nlp, input_raw_text)

    documents = [pp_text]

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
            top_term_ids = lda_model.get_topic_terms(topic[0], topn=20)
            top_terms = [lda_model.id2word[tup[0]] for tup in top_term_ids]
            relevant_topic_details.append((topic[0], top_terms))

        for word_tuple in doc_as_corpus:
            word_topics = lda_model.get_term_topics(word_tuple[0])
            if word_topics:
                output_word_topics.append((lda_model.id2word[word_tuple[0]], [wt[0] for wt in word_topics]))

    output_pred_label = classifier_model.predict(topic_vecs)[0]

    output_dict = {'pred_label': output_pred_label,
                   'overall_doc_topics': output_overall_topics,
                   'relevant_topic_terms': relevant_topic_details,
                   'per_word_topics': output_word_topics}

    return output_dict

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
        classifier = pickle.load(f)
    print("finished logistic regression model")

    print("loading spacy")
    nlp = spacy.load('en_core_web_md')
    print("finished loading spacy")

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

    pred_output_res = get_json_prediction_output(nlp=nlp, lda_model=lda, classifier_model=classifier, input_raw_text=DUMMY_TEXT)

    ### example usage below
    print("----------")
    print(DUMMY_TEXT)
    print("overall prediction: {}".format(pred_output_res['pred_label']))
    print("key topics relevant to input text: ---")
    topic_ids = []
    for terms in pred_output_res['relevant_topic_terms']:
        print(terms)
        topic_ids.append(terms[0])
    print("---")
    print("words matching topics in input text: ---")
    for id in topic_ids:
        relevant_words = []
        for word_topic in pred_output_res['per_word_topics']:
            if id in word_topic[1]:
                relevant_words.append(word_topic[0])
        print(id, ':', relevant_words)
    print("---")


if __name__ == "__main__":
    main()