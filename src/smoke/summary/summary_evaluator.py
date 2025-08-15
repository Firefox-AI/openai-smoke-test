import openai
import evaluate
import string
from ..unieval.metric.evaluator import get_evaluator
from ..unieval.utils import convert_to_json
from pydantic import BaseModel, Field
import instructor

class EvaluationScores(BaseModel):
    consistency: float = Field(..., description="Factual accuracy compared to the original content (0.0-1.0).")
    coherence: float = Field(..., description="Logical flow and readability of the summary (0.0-1.0).")
    fluency: float = Field(..., description="Grammar and style quality (0.0-1.0).")
    relevance: float = Field(..., description="How well the summary captures the main points (0.0-1.0).")
    overall: float = Field(..., description="A holistic score of the summary's quality (0.0-1.0).")

class SummaryEvaluator:
    def __init__(self, llm_scoring_config):
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

        self.unieval = get_evaluator("summarization", device="mps")
        self.unieval_dimension = ["consistency", "coherence", "fluency", "relevance"]

        self.llm_scoring_config = llm_scoring_config
        self.scorer_client = instructor.patch(openai.AsyncOpenAI(
            api_key=self.llm_scoring_config.get("api_key"), base_url=self.llm_scoring_config.get("base_url")
        ))

    def compute_unieval(self, prediction, reference, original_content):
        return self.unieval.evaluate(convert_to_json(
            output_list=[prediction],
            src_list=[original_content],
            ref_list=[reference],
        ), print_result=False, dims=self.unieval_dimension)


    async def score_summary_with_model(self, prediction, reference, original_content):
        return (await self.scorer_client.chat.completions.create(
            model=self.llm_scoring_config.get("model_name"),
            response_model=EvaluationScores,
            messages=[
                {"role": "system", "content": self.llm_scoring_config.get("system_prompt")},
                {"role": "user", "content": self.llm_scoring_config.get("user_prompt")
                        .replace("{original_content}", original_content)
                        .replace("{reference}", reference)
                        .replace("{prediction}", prediction)
                 },
            ],
            temperature=0.1,
            top_p=0.01,
            stream=False,
        )).model_dump()

    """Evaluates text that may contain duplicate or hallucinated characters/words

    Args:
        text: text to be evaluated

    Returns:
        bool: score is from 0 to 1 - 1 being 100% certain there's no hiccup (return 0)
        if score is below the is_hiccup_threshold, (return 1)
    """
    def contains_hiccup(self, text: str, threshold_ratio: float = 0.1, is_hiccup_threshold: float = 0.5) -> int:
        score = 1

        base_word_penalty = 0.05
        base_ngram_penalty = 0.10
        base_consecutive_penalty = 0.20

        # Pass 1 - normalize and count words
        words = text.split()
        if not words:
            return 0

        word_counts = {}
        for word in words:
            # Normalize the word by converting to lowercase and stripping punctuation
            normalized_word = word.lower().strip(string.punctuation)
            if normalized_word != "":
                word_counts[normalized_word] = word_counts.get(normalized_word, 0) + 1

        threshold = len(words) * threshold_ratio

        for word, count in word_counts.items():
            if count > threshold:
                # print(f"Pass 1: '{word}' repeated {count} times (+{base_word_penalty * (count / threshold)})")
                score -= base_word_penalty * (count - threshold)

        # Pass 2: Check for multi-word phrase hiccups (n-grams)
        # We can check for phrases of 2, 3, or even 4 words
        for n in range(2, 4):  # Check for 2-word and 3-word phrases
            phrase_counts = {}
            for i in range(len(words) - n + 1):
                # Build the normalized phrase
                phrase = " ".join([w.lower().strip(string.punctuation) for w in words[i:i + n]])
                if phrase:
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            # Check if any phrase count exceeds the threshold
            for count in phrase_counts.values():
                if count > threshold:
                    # print(f"Pass 2: {count} / {len(words)} (+{base_ngram_penalty * (count - threshold)})")
                    score -= base_ngram_penalty * (count - threshold)

        # Pass 3: Check for consecutive word repetition
        previous_word = ""
        for word in words:
            # Normalize the word just like in your other passes
            normalized_word = word.lower().strip(string.punctuation)
            if normalized_word and normalized_word == previous_word:
                # print(f"Pass 3: '{normalized_word}' repeated back-to-back. (+{base_consecutive_penalty}")
                score -= base_consecutive_penalty
            previous_word = normalized_word

        if min(1, score) < is_hiccup_threshold:
            return 1
        return 0

    async def score(self, predictions: str, reference_summary: str, text: str, score_with_llm: bool):
        return {
            "rouge": self.rouge.compute(predictions=[predictions], references=[reference_summary]),
            "bleu": self.bleu.compute(predictions=[predictions], references=[reference_summary]),
            "unieval": (await self.score_summary_with_model(predictions, reference_summary, text)) if score_with_llm else self.compute_unieval(predictions, reference_summary, text)[0],
            "hiccup": self.contains_hiccup(predictions),
        }
