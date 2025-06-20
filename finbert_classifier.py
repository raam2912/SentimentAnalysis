

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False
)


label_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

def classify_headline(headline: str) -> dict:
    """
    Classifies a financial news headline using FinBERT.

    Args:
        headline (str): The news headline to classify.

    Returns:
        dict: {
            "headline": str,
            "label": str,
            "score": float,  # model confidence
            "weighted_score": float  # +score for positive, -score for negative, 0 for neutral
        }
    """
    result = sentiment_pipeline(headline)[0]
    label = result['label'].lower()  # Normalize to lowercase
    score = round(result['score'], 4)
    weighted_score = round(label_map[label] * score, 4)

    return {
        "headline": headline,
        "label": label,
        "score": score,
        "weighted_score": weighted_score
    }
