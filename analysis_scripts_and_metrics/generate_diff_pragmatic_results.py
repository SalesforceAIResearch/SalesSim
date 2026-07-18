from typing import List, Dict
import json
import argparse
def aggregate_pragmatic_metrics(dialogue_feature_list: Dict[str, dict]) -> dict:
    """
    Given a dict oc onversation ID to features, aggregate the metrics into a single dict by mean. 
    {
    "20251016_232414_0dc073c3-5685-4e95-8df1-0fc8bf5f7c87": {
        "domain": "running_shoes",
        "pacing": 6,
        "complete_sentences": 0.375,
        "num_turns": 5,
        "avg_tfidf_similarity": 0.1084,
        "tfidf_turn_0_avg_similairty similarity": 0.0704,
        "tfidf_turn_1_avg_similairty similarity": 0.0,
        "tfidf_turn_2_avg_similairty similarity": 0.0366,
        "tfidf_turn_3_avg_similairty similarity": 0.0771
    },
    """
    if not dialogue_feature_list:
        return {}

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for feature_dict in dialogue_feature_list.values():
        if not isinstance(feature_dict, dict):
            continue

        for metric, value in feature_dict.items():
            # Aggregate only numeric metrics (e.g., pacing, num_turns, tfidf scores).
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[metric] = totals.get(metric, 0.0) + float(value)
                counts[metric] = counts.get(metric, 0) + 1

    return {metric: totals[metric] / counts[metric] for metric in totals if counts[metric] > 0}
def generate_diff_pragmatic_results(dialogue_feature_dict_to_compare: Dict[str, dict], crs_feature_dict: Dict[str, float]) -> dict:
    """
    Given a feature dicts, generate the diff in pragmatic features with the CRS dict
    """
    aggregated_feature_dict = aggregate_pragmatic_metrics(dialogue_feature_dict_to_compare)
    diff_pragmatic_results = {}
    for metric, value in aggregated_feature_dict.items():
        if metric not in crs_feature_dict:
            continue
        crs_value = crs_feature_dict[metric]
        diff = value - crs_value
        diff_pragmatic_results[metric] = diff
        print(f"{metric}: {diff}")
    return diff_pragmatic_results
    

# CLI to pass in list of dialogue_feature files 
def main(list_of_dialogue_feature_files_paths: List[str], crs_feature_dict: Dict[str, float]):
    for dialogue_feature_file_path in list_of_dialogue_feature_files_paths:
        dialogue_feature_dict = json.load(open(dialogue_feature_file_path))
        print(f"Calculating diff scores between {dialogue_feature_file_path} and CRS")
        generate_diff_pragmatic_results(dialogue_feature_dict, crs_feature_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diff pragmatic results")
    parser.add_argument("--list_of_dialogue_feature_files_paths", type=str, nargs="+", required=True)
    parser.add_argument("--path_to_dialogue_feature_to_compare_to", type=str, default="analysis_scripts_and_metrics/crs_dialogue_features.json")
    args = parser.parse_args()
    crs_dict = json.load(open(args.path_to_dialogue_feature_to_compare_to))
    crs_feature_dict = aggregate_pragmatic_metrics(crs_dict)
    main(args.list_of_dialogue_feature_files_paths, crs_feature_dict)