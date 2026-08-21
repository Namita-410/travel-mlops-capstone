"""
Hotel Recommendation Model
-----------------------------
Data note: in this dataset each `place` (city) maps to exactly one hotel,
so "recommend a hotel" is equivalent to "recommend a place this user hasn't
stayed in yet." The model is built and documented on that basis rather than
pretending there are multiple competing hotels per city.

Approach: hybrid of
  1. Item-based collaborative filtering -- places that tend to be visited
     together by the same users (cosine similarity over the user-place
     interaction matrix) surface likely-next places for a given user.
  2. Content-based price affinity -- a user's average nightly hotel spend
     is compared to each candidate place's average price, so recommendations
     stay in the user's typical budget band even for users with sparse
     history (cold-start-ish fallback).

Final score = 0.7 * collaborative_score + 0.3 * price_affinity_score
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "recommender.joblib")

COLLAB_WEIGHT = 0.7
PRICE_WEIGHT = 0.3


def build_artifacts():
    hotels = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))

    # user-place interaction matrix (implicit feedback = number of stays)
    interactions = hotels.groupby(["userCode", "place"]).size().unstack(fill_value=0)

    # item-item (place-place) similarity from co-occurrence patterns
    place_similarity = cosine_similarity(interactions.T)
    place_similarity_df = pd.DataFrame(
        place_similarity, index=interactions.columns, columns=interactions.columns
    )

    # per-place average nightly price, and per-user average nightly price paid
    place_avg_price = hotels.groupby("place")["price"].mean()
    user_avg_price = hotels.groupby("userCode")["price"].mean()

    scaler = MinMaxScaler()
    place_price_scaled = pd.Series(
        scaler.fit_transform(place_avg_price.values.reshape(-1, 1)).flatten(),
        index=place_avg_price.index,
    )

    return {
        "interactions": interactions,
        "place_similarity": place_similarity_df,
        "place_avg_price": place_avg_price,
        "place_price_scaled": place_price_scaled,
        "user_avg_price": user_avg_price,
        "all_places": list(interactions.columns),
    }


def recommend_for_user(artifacts, user_code: int, top_n: int = 3):
    interactions = artifacts["interactions"]
    place_similarity = artifacts["place_similarity"]
    place_price_scaled = artifacts["place_price_scaled"]
    user_avg_price = artifacts["user_avg_price"]
    all_places = artifacts["all_places"]

    if user_code not in interactions.index:
        # cold start: no history at all -> recommend the most broadly popular places
        popularity = interactions.sum(axis=0).sort_values(ascending=False)
        return list(popularity.head(top_n).index)

    user_visits = interactions.loc[user_code]
    visited_places = user_visits[user_visits > 0].index.tolist()
    unvisited_places = [p for p in all_places if p not in visited_places]

    if not unvisited_places:
        return []  # user has already stayed everywhere in the catalog

    # collaborative score: weighted similarity to places the user already likes
    collab_scores = {}
    for candidate in unvisited_places:
        score = sum(
            user_visits[visited] * place_similarity.loc[candidate, visited]
            for visited in visited_places
        )
        collab_scores[candidate] = score

    collab_series = pd.Series(collab_scores)
    if collab_series.max() > 0:
        collab_series = collab_series / collab_series.max()  # normalize to [0,1]

    # price-affinity score: 1 - distance between user's typical spend and place's price band
    user_price_norm = np.clip(
        (user_avg_price.get(user_code, place_price_scaled.mean()) - artifacts["place_avg_price"].min())
        / (artifacts["place_avg_price"].max() - artifacts["place_avg_price"].min() + 1e-9),
        0, 1,
    )
    price_affinity = {
        p: 1 - abs(user_price_norm - place_price_scaled[p]) for p in unvisited_places
    }
    price_series = pd.Series(price_affinity)

    final_score = COLLAB_WEIGHT * collab_series.reindex(unvisited_places).fillna(0) \
        + PRICE_WEIGHT * price_series.reindex(unvisited_places).fillna(0)

    return list(final_score.sort_values(ascending=False).head(top_n).index)


def main():
    artifacts = build_artifacts()
    joblib.dump(artifacts, MODEL_OUT)
    print(f"Saved recommender artifact -> {MODEL_OUT}")

    # sanity-check on a few real users
    sample_users = list(artifacts["interactions"].index[:5])
    for u in sample_users:
        visited = artifacts["interactions"].loc[u]
        visited = visited[visited > 0].index.tolist()
        recs = recommend_for_user(artifacts, u, top_n=3)
        print(f"\nUser {u} -- visited {len(visited)}/{len(artifacts['all_places'])} places")
        print(f"  Already visited: {visited}")
        print(f"  Recommended:     {recs}")


if __name__ == "__main__":
    main()
