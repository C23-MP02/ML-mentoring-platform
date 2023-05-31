from function.preprocessing import *
import pandas as pd
import numpy as np
from collections import OrderedDict

def rank_sim(data):
    mentee_interest, mentors_interest, mentors_rating = transform_data(data)
    similarity_rank = {}
    score_per_mentor = {}
    mentee_id = mentee_interest['id']
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
        sim = cosine_similarity(interest_vec_mentee, interest_vec_mentor)
        if np.isnan(sim):
            sim = 1
        normalized_rating = mentors_rating.loc[mentors_rating['id']==mentor, "average_rating"].values[0] / 5
        score = 0.6 * sim + 0.4 * normalized_rating
        score_per_mentor[int(mentor)] = score

    similarity_rank[int(mentors_id[0])] = score_per_mentor

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