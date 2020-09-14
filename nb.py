import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def word_feats(words):
    return dict([(word, True) for word in words])


if __name__ == "__main__":
    negative_ids = movie_reviews.fileids('neg')
    positive_ids = movie_reviews.fileids('pos')

    negative_features = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negative_ids]
    positive_features = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in positive_ids]

    neg_cutoff = len(negative_features) * 3 / 4
    pos_cutoff = len(positive_features) * 3 / 4

    training_features = negative_features[:neg_cutoff] + positive_features[:pos_cutoff]
    testing_features = negative_features[neg_cutoff:] + positive_features[pos_cutoff:]
    print('train on %d instances, test on %d instances' % (len(training_features), len(testing_features)))

    classifier = NaiveBayesClassifier.train(training_features)
    print('accuracy:', nltk.classify.util.accuracy(classifier, testing_features))

    classifier.show_most_informative_features()
