from function.preprocessing import *
import pandas as pd
import numpy as np
from collections import OrderedDict

def rank_sim(data):
    mentee_interest, mentors_interest, mentors_rating = transform_data(data)
    similarity_rank = {}
    score_per_mentor = {}
    mentee_id = mentee_interest['id']
    mentee_gender = mentee_interest['gender_id'][0]
    mentors_id = list(mentors_interest['id'].unique())
    interest_vars = ['is_path_android', 'is_path_web', 'is_path_ios', 'is_path_ml', 'is_path_flutter',
                'is_path_fe', 'is_path_be', 'is_path_react', 'is_path_devops', 'is_path_gcp']
    
    interest_vec_mentee = mentee_interest.loc[mentee_interest['id']==mentee_id, interest_vars].values.reshape(-1,1)
    interest_vec_mentee = np.squeeze(np.asarray(interest_vec_mentee))
    interest_vec_mentee = [int(element) for element in interest_vec_mentee]
    for mentor in mentors_id:
        interest_vec_mentor = mentors_interest.loc[mentors_interest['id']==mentor, interest_vars].values.reshape(-1,1)
        interest_vec_mentor = np.squeeze(np.asarray(interest_vec_mentor))
        interest_vec_mentor = [int(element) for element in interest_vec_mentor]
        mentor_gender = mentors_interest.loc[mentors_interest['id']==mentor, "gender_id"].values[0]

        sim = modified_cosine_similarity(interest_vec_mentee, interest_vec_mentor)
        if np.isnan(sim):
            sim = 0
        rating = mentors_rating.loc[mentors_rating['id']==mentor, "average_rating"].values[0]
        if rating is None:
            rating = 0
        else:
            if np.isnan(rating):
                rating = 0
        normalized_rating = rating / 5

        gender_weight = 0
        if mentee_gender == 2:
            if mentee_gender == mentor_gender:
                gender_weight = 0.1

        score = 0.55 * sim + 0.35 * normalized_rating + gender_weight
        score_per_mentor[str(mentor)] = score

    similarity_rank[str(mentee_id[0])] = score_per_mentor

    # sorted similarity dictionary
    ranked_similarity_rank = {}
    for dict_keys, dict_vals in similarity_rank.items():
        ranked_dict_vals = {k: v for k, v in sorted(dict_vals.items(), key=lambda item: item[1], reverse=True)}
        ranked_similarity_rank[dict_keys] = ranked_dict_vals

    # output similarity dictionary
    ranked_sim_rank = {}
    for dict_keys, dict_vals in ranked_similarity_rank.items():
        list_mentors = list(dict_vals.keys())
        ranked_sim_rank[dict_keys] = list_mentors

    return ranked_sim_rank
    # return ranked_similarity_rank, ranked_sim_rank