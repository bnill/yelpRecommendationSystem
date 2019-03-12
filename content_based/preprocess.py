import os

def clean_review_data(reviews_pd):
  """
  remove reviews that are not rated
  """
  reviews_pd = reviews_pd.drop(reviews_pd[reviews_pd['stars'] == 0].index)
  # reviews_pd.drop_duplicates(['review_id'], keep='last', inplace=True)

  return reviews_pd


def get_reviews_conditions(reviews_pd, users_pd, businesses_pd):
  df = reviews_pd.set_index('business_id').join(businesses_pd.set_index('business_id'), rsuffix='__business').reset_index()
  df = df.drop(df[df['city'] == 'Las Vegas'].index)
  df = df.drop(df[df['review_count'] < 500].index)

  df = df.set_index('user_id').join(users_pd.set_index('user_id'),  rsuffix='__user').reset_index()
  df = df.drop(df[df['review_count__user'] < 50].index)

  return df


def get_reviews_data_count_users_businesses(reviews_pd, count=50):
  reviews_pd_cleaned = clean_review_data(reviews_pd)
  
  while(True):
    tips_gby_business_count_user = reviews_pd_cleaned.groupby('business_id')['user_id'].count()
    tips_gby_user_count_business = reviews_pd_cleaned.groupby('user_id')['business_id'].count()

    tips_reviews__50_user = reviews_pd_cleaned.drop(reviews_pd_cleaned[reviews_pd_cleaned['user_id'].isin(tips_gby_user_count_business[tips_gby_user_count_business < count].index)].index)
    tips_reviews__50_user_50_busi = tips_reviews__50_user.drop(tips_reviews__50_user[tips_reviews__50_user['business_id'].isin(tips_gby_business_count_user[tips_gby_business_count_user < count].index)].index)

    if tips_reviews__50_user_50_busi.shape == reviews_pd_cleaned.shape:
      break

    reviews_pd_cleaned = tips_reviews__50_user_50_busi

  return reviews_pd_cleaned



def get_tips_reviews_data_without_null(tips_pd, reviews_pd):
  """
  left join tips-reviews datasets and remove all the rows that have null star ratings

  Returns: dataset with index ['usr_id', 'business_id']
  """
  print('cleaning reviews_pd')
  reviews_pd = clean_review_data(reviews_pd)
  tips_reviews_pd = tips_pd.set_index(['business_id', 'user_id']).join(reviews_pd.set_index(['business_id', 'user_id']), how='left', rsuffix="_review")
  
  print('cleaning tips_reviews_pd')
  tips_reviews_pd_cleaned = tips_reviews_pd.drop(tips_reviews_pd[tips_reviews_pd['stars'].isnull()].index).reset_index()
  tips_reviews_pd_cleaned.drop_duplicates(['review_id', 'user_id', 'business_id'], keep='last', inplace=True)
  return tips_reviews_pd_cleaned


def get_tips_reviews_data_count_users_businesses(tips_pd, reviews_pd, count=50):
  """
  left join tips-reviews datasets and remove all the rows that have null star ratings
  remove users, businesses who does not have at least 50 ratings

  tips_pd:
  reviews_pd
  count: number of reviews per user and business to consider

  Returns: dataset with users and businesses who have done at least 50 reviews
  """
  tips_reviews_pd_cleaned = get_tips_reviews_data_without_null(tips_pd, reviews_pd)

  while(True):
    tips_gby_business_count_user = tips_reviews_pd_cleaned.groupby('business_id')['user_id'].count()
    tips_gby_user_count_business = tips_reviews_pd_cleaned.groupby('user_id')['business_id'].count()

    tips_reviews__50_user = tips_reviews_pd_cleaned.drop(tips_reviews_pd_cleaned[tips_reviews_pd_cleaned['user_id'].isin(tips_gby_user_count_business[tips_gby_user_count_business < count].index)].index)
    tips_reviews__50_user_50_busi = tips_reviews__50_user.drop(tips_reviews__50_user[tips_reviews__50_user['business_id'].isin(tips_gby_business_count_user[tips_gby_business_count_user < count].index)].index)

    if tips_reviews__50_user_50_busi.shape == tips_reviews_pd_cleaned.shape:
      break

    tips_reviews_pd_cleaned = tips_reviews__50_user_50_busi

  return tips_reviews_pd_cleaned


