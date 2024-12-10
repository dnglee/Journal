from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.utils import TRANSFORMERS_CACHE

# Load the tokenizer and model
model_name = "bhadresh-savani/bert-base-uncased-emotion"  # Fine-tuned on GoEmotions
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# List of emotions
emotions = list(config.id2label.values())

async def get_emotions(journal_entry):
    # Tokenize the text
    inputs = tokenizer(journal_entry, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=1)

    # Get the top predictions
    top_emotions = torch.topk(probs, k=3)  # Top 3 emotions
    threshold = 0.1

    filter_emotions = [
        {"emotion": emotions[idx], "score": score.item()}
        for idx, score in zip(top_emotions.indices[0], top_emotions.values[0])
        if score.item() > threshold
    ]

    return filter_emotions