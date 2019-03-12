
from math import sqrt
from preprocess import *
from analyze_datasets import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_distances 
from sklearn.metrics import mean_squared_error

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import ngrams
from nltk.corpus import stopwords


lmtzr = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))


# TODO handle multiple review ids

def get_tokens(review):
  """
  Takes input a review, perform lemmatization, find 2 grams, remove stopwords and returns tokens as a list
  """
  tokens = nltk.word_tokenize(review)
  tokens = [lmtzr.lemmatize(token) for token in tokens]

  """
  tokens_2grams = ngrams(tokens, 2)

  tokens = [token for token in tokens if token not in stopwords_set]
  tokens_2grams = [" ".join(tokens_2gram) for tokens_2gram in tokens_2grams]

  tokens.extend(tokens_2grams)
  """

  return tokens


def get_train_test(reviews_file, tips_file, count, use_tips=False, save=False, loaded=True):
  if loaded:
    return load_train_test()

  reviews_pd = read_data_review(reviews_file) 
  if use_tips:
    tips_pd = read_data_tips(tips_file)
    dataset_cleaned = get_tips_reviews_data_without_null(tips_pd, reviews_pd)
  else:
    dataset_cleaned = get_reviews_data_count_users_businesses(reviews_pd, count=count)


  """
  # for every user, business pair, 25% reviews in test set
  # return train_test_split(dataset_cleaned, test_size=0.25, random_state=7, stratify=None)
  random_state = np.random.RandomState(seed=7)
  sample_fn = lambda obj: obj.loc[random_state.choice(obj.index, int(obj.shape[0]/4), replace=False), :]
  # test_dataset = dataset_cleaned.groupby(['business_id', 'user_id']).apply(sample_fn)
  test_dataset = dataset_cleaned.groupby(['business_id', 'user_id']).apply(lambda x: x.sample(frac=0.25, random_state=7))
  # test_dataset.reset_index(inplace=True, drop=True)
  train, test = dataset_cleaned.drop(dataset_cleaned[dataset_cleaned['review_id'].isin(test_dataset['review_id'])].index).reset_index(), test_dataset.reset_index(drop=True)
  """

  train, test = train_test_split(dataset_cleaned, random_state=7)

  if save:
    train.to_csv('/scratch/raval.v/yelp_dataset/train.csv')
    test.to_csv('/scratch/raval.v/yelp_dataset/test.csv')

  return train, test


def get_train_test_conditioned(reviews_file):
  """
  reviews_file is csv file obtained using get_reviews_conditions 
  """
  train, test = train_test_split(pd.read_csv(reviews_file), random_state=7)
  train.to_csv('/scratch/raval.v/yelp_dataset/train_conditioned.csv')
  test.to_csv('/scratch/raval.v/yelp_dataset/test_conditioned.csv')

  return train, test



def load_train_test():
  train = pd.read_csv('/scratch/raval.v/yelp_dataset/train_conditioned.csv')
  test = pd.read_csv('/scratch/raval.v/yelp_dataset/test_conditioned.csv')
  return train, test


class Lda():

  def __init__(self, reviews_file='/scratch/raval.v/yelp_dataset/yelp_academic_dataset_review_try.json',
                tips_file='/scratch/raval.v/yelp_dataset/yelp_academic_dataset_tip_try.json', use_tips=False, count=20):
    self.reviews_file = reviews_file
    self.tips_file = tips_file
    self.use_tips = use_tips
    self.count = count


  def make_document_term(self, reviews, fit=True):
    """
    reviews: a list-like object that can be reviews or tips

    Returns: document-term matrix
    """
    if fit:
      self.vec = CountVectorizer(tokenizer=get_tokens, stop_words='english', ngram_range=(1, 2))
      self.vec.fit(reviews)

    return self.vec.transform(reviews)

 
  def run_lda(self, doc_terms_all_stars, fit=True, n_topics=20):
    """
    gets features from reviews conditioned on star rating and
    finds distribution of reviews over words
    """
    # list of document topics distribution for each star review
    doc_topics = []
    doc_words = []
    if fit:
      print('going to fit lda')
      self.ldas = [LatentDirichletAllocation(n_components=n_topics, verbose=10, n_jobs=-1, max_iter=10)]*len(doc_terms_all_stars)
      for i, doc_term in enumerate(doc_terms_all_stars):
        print('fitting, transforming lda for {}th star'.format(i))
        # doc_topics.append(self.ldas[i].fit_transform(doc_term))
        doc_topic = self.ldas[i].fit_transform(doc_term)
        doc_words.append(np.dot(doc_topic, self.ldas[i].components_))
    else:
      print('going to transform lda')
      for i, doc_term in enumerate(doc_terms_all_stars):
        print('transforming for {}th star'.format(i))
        # doc_topics.append(self.ldas[i].transform(doc_term))
        doc_topic = self.ldas[i].transform(doc_term)
        doc_words.append(np.dot(doc_topic, self.ldas[i].components_))

    return doc_words


  def get_lda_features(self, dataset, fit=True):
    """
    runs lda and gets features in form of pandas dataframe

    dataset: cleaned dataset
    returns: review ids of the users, list of doc_topic (5 elements)
    """
    # list of indices
    reviews_all_stars_index = []

    # document term matrix
    doc_terms_all_stars = []

    print('preprocessing for each star, order is {}'.format(dataset['stars'].unique()))
    reviews_all = dataset['text']
    doc_term = self.make_document_term(reviews_all.tolist(), fit=fit)
    for star in dataset['stars'].unique():
      reviews_stars_index = dataset[dataset['stars'] == star].index
      reviews_all_stars_index.extend(reviews_stars_index)
      doc_terms_all_stars.append(doc_term[reviews_stars_index.values])

    reviews_all_stars_index = pd.Index(reviews_all_stars_index)

    doc_words = self.run_lda(doc_terms_all_stars, fit=fit)
    doc_words_flat = []
    for doc_word in doc_words:
      doc_words_flat.extend(doc_word)

    # make data frame consisting of all the features, followed by dataset columns names
    df = pd.DataFrame(doc_words_flat)
    df = df.join(dataset.loc[reviews_all_stars_index, :])
    return df


  def form_user_features(self, dataset):
    """
    dataset: dataset as returned by get_lda_features
    """
    columns = list(range(dataset.shape[1]-12))
    columns.append('user_id')
    users = dataset.loc[:, columns].groupby(['user_id']).mean()
    return users


  def form_business_features(self, dataset):
    """
    same thing as form_user_features but with business
    """
    columns = list(range(dataset.shape[1]-12))
    columns.append('business_id')
    businesses = dataset.loc[:, columns].groupby(['business_id']).mean()
    return businesses


  def find_similarity_rating(self, users, items):
    """
    finds cosines similarity and calculates ratings using user-item feature pair
    users: list of users with their features
    items: list of items with their features
    """
    return paired_distances(users, items, metric='cosine')*5


  def item_clustering(self, X):
    """
    perform item clustering

    X: feature matrix
    """
    pass


  def calc_error_metrics(self, y_test, y_pred):
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print('root mean squared error is {}'.format(rmse))


  def train_test_model(self):
    """
    divides dataset to train and test and finds model performance
    """
    print('getting train, test datasets')
    # train_dataset, test_dataset = get_train_test(self.reviews_file, self.tips_file, self.count,
    #                                               use_tips=self.use_tips, loaded=True, save=False)
    train_dataset, test_dataset = load_train_test()

    train_dataset.reset_index(inplace=True)
    test_dataset.reset_index(inplace=True)

    print('getting lda features using train set')
    train_reviews_X = self.get_lda_features(train_dataset, fit=True)

    print('calculating user and business features using returned lda results')
    users = self.form_user_features(train_reviews_X)
    businesses = self.form_business_features(train_reviews_X)

    datasets = [train_dataset, test_dataset]
    dataset_type = ['train dataset', 'test dataset']

    # choose only user ids and business ids that are in training set
    for i, dataset in enumerate(datasets):
      dataset = dataset[dataset['user_id'].isin(users.index)]
      dataset = dataset[dataset['business_id'].isin(businesses.index)]

      # get users and businesses to find similarity
      test_user_ids = dataset.loc[:, 'user_id']
      test_business_ids = dataset.loc[:, 'business_id']

      # get vectors of all the users and businesses
      test_user_X = users.loc[test_user_ids]
      test_business_X = businesses.loc[test_business_ids]

      print('finding similarity between users, businesses on {}'.format(dataset_type[i]))
      y_pred = self.find_similarity_rating(test_user_X, test_business_X)
      y_test = dataset.loc[:, 'stars']

      print('calculating error metrics')
      self.calc_error_metrics(y_test, y_pred)


if __name__ == '__main__':
  lda = Lda()
  lda.train_test_model()
