def make_preference_reward(scorer):
    """
    Factory that returns a GRPO-compatible reward function.

    Scores each generated completion by ROUGE-L similarity to the chosen
    (high-quality) response, scaled by its GPT-4 quality rating.

    - ROUGE-L captures longest common subsequence overlap — a proxy for
      whether the generated response covers the same content as the chosen one.
    - chosen_rating (1–5) scales the reward: prompts where even the best
      response scored 3.0 contribute less signal than prompts with a 5.0
      gold response. Normalized to [0.5, 1.0] so it never zeroes out reward.
    """

    def preference_reward(completions, chosen, chosen_rating, **kwargs):
        rewards = []
        for completion, ref, rating in zip(completions, chosen, chosen_rating):
            gen_text = completion[0]["content"] if isinstance(completion, list) else completion

            # chosen is a list of messages — extract assistant turn(s)
            if isinstance(ref, list):
                ref_text = " ".join(
                    m["content"] for m in ref if isinstance(m, dict) and m.get("role") == "assistant"
                )
            else:
                ref_text = str(ref)

            rouge_l = scorer.score(gen_text, ref_text)["rougeL"].fmeasure

            # Map rating 1–5 → quality_scale 0.5–1.0
            quality_scale = 0.5 + (float(rating) - 1.0) / 8.0

            rewards.append(rouge_l * quality_scale)
        return rewards

    return preference_reward
