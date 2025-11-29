from collections import Counter
import re
def get_mode_score(scores):
    if not scores:
        return None
    score_counts = Counter(scores)
    most_common = score_counts.most_common()
    if most_common:
        # If multiple modes, pick the smallest one
        max_count = most_common[0][1]
        modes = [val for val, count in most_common if count == max_count]
        return min(modes)
    return None
def extract_scores(feedback):
    feedback_text = "\n".join(str(f) for f in feedback)
    # Find all <score>...</score> tags
    scores = re.findall(r"<score>(.*?)</score>", feedback_text, re.DOTALL)
    # Try to convert to float or int
    parsed_scores = []
    for s in scores:
        try:
            parsed_scores.append(float(s.strip()))
        except Exception:
            continue
    return parsed_scores

def get_big5_scores(votes, trait_name):
    vote_counts = Counter(votes)
    most_common_vote = vote_counts.most_common(1)[0][0]
    if trait_name is not "Neuroticism":
        if "High" in vote_counts: # Correct for false negative for high.
            most_common_vote = "High"
        if "Low" in vote_counts:
            most_common_vote = "Low" 
    return most_common_vote

def convert_big5_level_to_value(level):
    if level == "High":
        return 1
    elif level == "Neutral":
        return 0.5
    elif level == "Low":
        return 0
    raise ValueError(f"Invalid big5 level: {level}")

def aggregate_float_scores(results, dimension):
    scores = []
    for result in results:
        dimension_scores = result.get("dimension_scores", {})
        if dimension not in dimension_scores:
            continue
        score = dimension_scores[dimension]
        scores.append(score)

    mean = sum(scores) / len(scores)
    return mean

def aggregate_big5_scores(results, big5_trait, conversations):
    scores = []
    for idx, result in enumerate(results):
        conv = conversations[idx]
        dimension_scores = result.get("dimension_scores", {})
        if big5_trait not in dimension_scores:
            continue
        gold_score = convert_big5_level_to_value(conv.get("shopper_big5_traits", {}).get(big5_trait.split("_")[-1].lower()))
        score = convert_big5_level_to_value(dimension_scores[big5_trait])
        scores.append(1 - abs(gold_score - score)) # 1.0 is best score. 
    mean = sum(scores) / len(scores)
    return mean
