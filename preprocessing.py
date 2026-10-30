
import re
import string
from typing import Dict, List, Any

# Import spaCy for NLP processing
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to a blank English pipeline if model is not yet downloaded
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None


def clean_text(text: str) -> str:
    """
    Clean raw input text:
    - Checks for empty input
    - Converts text to lowercase
    - Normalizes extra spaces and newlines
    - Removes unnecessary special characters while preserving symbols
      meaningful in technical terms (like c++, c#, .net, scikit-learn).
    """
    if not text or not text.strip():
        return ""

    # 1. Convert to lowercase
    cleaned = text.lower()

    # 2. Standardize bullet points and dashes
    cleaned = re.sub(r'[\u2022\u2023\u25e6\u2043\u2219\u25aa\u25cf\u25cb\u25a0*•-]', ' ', cleaned)

    # 3. Replace non-breaking spaces and tabs with standard space
    cleaned = cleaned.replace('\u00a0', ' ').replace('\t', ' ')

    # 4. Remove unnecessary punctuation/special characters,
    # but carefully preserve technical characters: +, #, ., /
    cleaned = re.sub(r'[^a-z0-9\s+#./-]', ' ', cleaned)

    # 5. Collapse multiple whitespace and newlines into single spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned


def tokenize_and_lemmatize(text: str) -> Dict[str, List[str]]:
    """
    Tokenize text and apply lemmatization:
    - Splits text into meaningful word tokens
    - Removes English stop words (like 'is', 'the', 'at', 'which')
    - Converts words to their root forms (e.g. "developing", "developed" -> "develop")

    Returns a dictionary containing:
      - 'tokens': all meaningful word tokens
      - 'lemmas': lemmatized root forms of tokens
      - 'filtered_tokens': stop-word removed, lemmatized tokens
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"tokens": [], "lemmas": [], "filtered_tokens": []}

    tokens = []
    lemmas = []
    filtered_tokens = []

    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(cleaned)
        for token in doc:
            token_text = token.text
            if not token_text.strip():
                continue

            tokens.append(token_text)
            lemma = token.lemma_.lower()
            lemmas.append(lemma)

            # Keep token if it's not a stop word and not standalone punctuation
            if not token.is_stop and token_text not in string.punctuation:
                if len(lemma) > 1 or lemma in {'c', 'r'}:
                    filtered_tokens.append(lemma)
    else:
        # Fallback pure-Python tokenizer if spaCy is not installed
        basic_stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
            "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
            "its", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
            "on", "once", "only", "or", "other", "our", "out", "over", "own", "same", "she",
            "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
            "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
            "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
            "while", "who", "whom", "why", "with", "you", "your"
        }
        raw_words = cleaned.split()
        for w in raw_words:
            w_clean = w.strip(string.punctuation)
            if not w_clean:
                continue
            tokens.append(w_clean)
            lemma = w_clean
            if lemma.endswith("ing") and len(lemma) > 5:
                lemma = lemma[:-3]
            elif lemma.endswith("ed") and len(lemma) > 4:
                lemma = lemma[:-2]
            elif lemma.endswith("s") and len(lemma) > 3 and not lemma.endswith("ss"):
                lemma = lemma[:-1]
            lemmas.append(lemma)

            if w_clean not in basic_stopwords:
                filtered_tokens.append(lemma)

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "filtered_tokens": filtered_tokens,
    }


def extract_named_entities(raw_text: str) -> Dict[str, List[str]]:
    """
    Extract Named Entities (names, organizations, locations, dates) using spaCy.
    Takes original raw text (with original casing) for best NER accuracy.
    """
    if not raw_text or not raw_text.strip():
        return {}

    entities: Dict[str, List[str]] = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],
        "DATE": []
    }

    if SPACY_AVAILABLE and nlp is not None and "ner" in nlp.pipe_names:
        doc = nlp(raw_text)
        for ent in doc.ents:
            label = ent.label_
            text_val = ent.text.strip().replace("\n", " ")
            if label in entities and text_val not in entities[label]:
                entities[label].append(text_val)

    return entities
