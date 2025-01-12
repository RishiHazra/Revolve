import glob
import os
import sys

sys.path.append(os.environ["ROOT_PATH"])
from collections import Counter
import argparse
import pandas as pd
from utils import *
from rewards_database import RewardsDatabase


def update_elo(rating1, rating2, result):
    # Basic parameters for the Elo rating system
    K = 32  # Maximum change per game
    # Calculate expected scores
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
    # Update ratings
    new_rating1 = rating1 + K * (result - expected1)
    new_rating2 = rating2 + K * (1 - result - expected2)
    return new_rating1, new_rating2


def elo_scores(df):
    ratings = {
        video: 1500 for video in pd.concat([df["Video 1"], df["Video 2"]]).unique()
    }
    # Initialize ratings
    for index, row in df.iterrows():
        video1, video2, selected = row["Video 1"], row["Video 2"], row["Selected"]
        if selected == 1.0:
            result = 1.0  # Video_1 wins
        elif selected == 2.0:
            result = 0.0  # Video_2 wins
        else:
            result = 0.5  # Tie

        new_elo1, new_elo2 = update_elo(ratings[video1], ratings[video2], result)
        ratings[video1], ratings[video2] = new_elo1, new_elo2

    # Normalize ratings to range [0, 1]
    max_rating = max(ratings.values())
    min_rating = min(ratings.values())
    normalized_ratings = {
        k: (v - min_rating) / (max_rating - min_rating) for k, v in ratings.items()
    }

    return normalized_ratings


def group_feedback(df):
    def split_and_add(feedback_list):
        try:
            return [feedback for feedback in feedback_list.split(", ")]
        except AttributeError:
            return []

    all_videos = set(df["Video 1"].tolist()).union(df["Video 2"].tolist())
    feedback_dict = {
        k: {"Positive Feedback": [], "Negative Feedback": []} for k in all_videos
    }
    for _, row in df.iterrows():
        video_1, pos_feedback_1, neg_feedback_1 = (
            row["Video 1"],
            row["Positive Feedback 1"],
            row["Negative Feedback 1"],
        )
        video_2, pos_feedback_2, neg_feedback_2 = (
            row["Video 2"],
            row["Positive Feedback 2"],
            row["Negative Feedback 2"],
        )
        feedback_dict[video_1]["Positive Feedback"].extend(
            split_and_add(pos_feedback_1)
        )
        feedback_dict[video_1]["Negative Feedback"].extend(
            split_and_add(neg_feedback_1)
        )
        feedback_dict[video_2]["Positive Feedback"].extend(
            split_and_add(pos_feedback_2)
        )
        feedback_dict[video_2]["Negative Feedback"].extend(
            split_and_add(neg_feedback_2)
        )

    for k, v in feedback_dict.items():
        # save only the most common positive and negative feedback
        all_pos_counter = Counter(v["Positive Feedback"])
        all_pos = [
            k for k, _ in all_pos_counter.most_common(min(len(all_pos_counter), 2))
        ]
        all_neg_counter = Counter(v["Negative Feedback"])
        all_neg = [
            k for k, _ in all_neg_counter.most_common(min(len(all_neg_counter), 2))
        ]
        # remove intersections
        intersection = list(set(all_pos).intersection(set(all_neg)))
        all_pos = [elem for elem in all_pos if elem not in intersection]
        all_neg = [elem for elem in all_neg if elem not in intersection]
        feedback_dict[k] = {"Positive Feedback": all_pos, "Negative Feedback": all_neg}

    feedback_df = pd.DataFrame.from_dict(feedback_dict, orient="index")
    feedback_df.index.name = "Video"
    return feedback_df


# Example usage:
def main():
    parser = argparse.ArgumentParser(description="Run the Video Survey Application.")
    parser.add_argument(
        "--gen_id", required=True, type=int, help="Run the application in dummy mode"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=str,
        choices=["hf_database", "hf_eureka_database"],
        help="name of database?",
    )
    args = parser.parse_args()
    GEN_ID = args.gen_id
    baseline = args.baseline
    response_filename = f"responses_*.csv"
    load_dir = os.path.join(
        os.environ["ROOT_PATH"], "human_feedback", f"generation_{GEN_ID}"
    )
    response_paths = glob.glob(f"{load_dir}/{response_filename}")
    # also load responses from prev gens
    if GEN_ID > 0:
        for id in range(GEN_ID):
            load_dir = os.path.join(
                os.environ["ROOT_PATH"], "human_feedback", f"generation_{id}"
            )
            response_paths += glob.glob(f"{load_dir}/{response_filename}")
    df = pd.DataFrame(
        columns=[
            "Video 1",
            "Video 2",
            "Selected",
            "Positive Feedback 1",
            "Negative Feedback 1",
            "Positive Feedback 2",
            "Negative Feedback 2",
        ]
    )
    for response_path in response_paths:
        df_response = pd.read_csv(response_path)
        df = pd.concat([df, df_response], ignore_index=True)

    scores = elo_scores(df)
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(scores)
    scores_df = pd.DataFrame.from_dict(scores)
    scores_df.columns = ["Video", "Score"]
    feedback_df = group_feedback(df)
    combined_df = pd.merge(scores_df, feedback_df, on="Video")

    model_name = "gpt-4"
    max_group_size = 8
    num_groups = 14
    crossover_prob = 0.5
    # it = args.gen_id

    for _, row in combined_df.iterrows():
        path, hf_fitness_score, pos_feedback, neg_feedback = (
            row["Video"],
            row["Score"],
            ", ".join(row["Positive Feedback"]),
            ", ".join(row["Negative Feedback"]),
        )
        reg_str = "\\" if "\\" in path else "/"
        group_id, it, running_number = [
            int(x)
            for x in path.split(reg_str)[-1]
            .replace(".mp4", "")
            .replace("gr", "")
            .split("_")
        ]
        # assert it == GEN_ID, "bug in saving hf fitness score"
        database_dir = os.path.join(
            os.environ["ROOT_PATH"], f"hf_database/{model_name}/group_{group_id}"
        )
        filename_suffix = f"{it}_{running_number}"
        reward_fn_filename = os.path.join(
            database_dir, f"reward_fns/{filename_suffix}.txt"
        )
        save_fitness_score(
            hf_fitness_score, model_name, group_id, it, running_number, baseline
        )
        human_feedback = f"{pos_feedback}\n{neg_feedback}"
        save_human_feedback(
            human_feedback, model_name, group_id, it, running_number, baseline
        )
        if it > 0:
            # now load groups
            rewards_database = RewardsDatabase(
                num_groups=num_groups,
                max_size=max_group_size,
                crossover_prob=crossover_prob,
                model_name=model_name,
                load_groups=False if GEN_ID == 0 else True,
                baseline=baseline,
            )
            # next add reward to group (only for current generation)
            if it == GEN_ID:
                rewards_database.add_reward_to_group(
                    [reward_fn_filename], [hf_fitness_score], [group_id]
                )


if __name__ == "__main__":
    main()
