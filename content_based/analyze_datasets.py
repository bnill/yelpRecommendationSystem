import os
import json
import pandas as pd
import numpy as np

from preprocess import * 

def read_data_user(file = '/scratch/raval.v/yelp_dataset/yelp_academic_dataset_user.json'):
  users_pd = pd.read_json(file, lines=True)
  return users_pd


def read_data_business(file = '/scratch/raval.v/yelp_dataset/yelp_academic_dataset_business.json'):
  businesses_pd = pd.read_json(file, lines=True)
  return businesses_pd


def read_data_review(file='/scratch/raval.v/yelp_dataset/yelp_academic_dataset_review.json'):
  # reviews_pd = pd.read_json(file, lines=True)
  print('loading reviews data from file {}'.format(file))
  reviews_list = []
  for line in open(file):
    reviews_list.append(json.loads(line.strip()))

  reviews_pd = pd.DataFrame(reviews_list)
  return reviews_pd


def read_data_tips(file='/scratch/raval.v/yelp_dataset/yelp_academic_dataset_tip.json'):
  # tips_pd = pd.read_json(file, lines=True)
  print('loading tips data from file {}'.format(file))
  tips_list = []
  for line in open(file):
    tips_list.append(json.loads(line.strip()))
  
  tips_pd = pd.DataFrame(tips_list)
  return tips_pd


def desc_tips_having_ratings(tips_pd, reviews_pd):
  tips_reviews_pd_cleaned = get_tips_reviews_data_without_null(tips_pd, reviews_pd)
  print('total number of tips having star ratings are {}'.format(tips_reviews_pd_cleaned.shape[0])) 
  print('number of tips with corresponding ratings')
  print(tips_reviews_pd_cleaned['stars'].value_counts()) 

  tips_gby_user_count_business = tips_reviews_pd_cleaned.groupby('user_id')['business_id'].count()

  counts = [1, 2, 10, 20, 50]
  for count in counts:
    ans_count = tips_gby_user_count_business[tips_gby_user_count_business >= count].shape[0]
    total_reviews = tips_gby_user_count_business[tips_gby_user_count_business >= count].sum()
    print('number of users with more than or equal to {} ratings is {} ({} reviews)'.format(count, ans_count, total_reviews))

  tips_gby_business_count_user = tips_reviews_pd_cleaned.groupby('business_id')['user_id'].count()
  for count in counts:
    ans_count = tips_gby_business_count_user[tips_gby_business_count_user >= count].shape[0]
    total_reviews = tips_gby_business_count_user[tips_gby_business_count_user >= count].sum()
    print('number of businesses with more than or equal to {} ratings is {} ({} reviews)'.format(count, ans_count, total_reviews))

  for count in counts:
    tips_reviews__50_user = tips_reviews_pd_cleaned.drop(tips_reviews_pd_cleaned[tips_reviews_pd_cleaned['user_id'].isin(tips_gby_user_count_business[tips_gby_user_count_business < count].index)].index)
    tips_reviews__50_user_50_busi = tips_reviews__50_user.drop(tips_reviews__50_user[tips_reviews__50_user['business_id'].isin(tips_gby_business_count_user[tips_gby_business_count_user < count].index)].index)
    print('number of reviews and tips rated by {} users and for {} businesses are {}'.format(count, count, tips_reviews__50_user_50_busi.shape[0])) 



if __name__ == '__main__':
  pass
