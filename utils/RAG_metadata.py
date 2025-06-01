# RAG_metadata.py

"""
RAG_metadata.py

Utilities for extracting semantic metadata from medical texts / queries using spaCy:
- Named entities: diseases, symptoms, procedures
- Age/gender categorization
- Noun phrase keyword extraction

Used for metadata-based filtering in the hybrid retriever.
"""


from pathlib import Path

import spacy
import re
from collections import Counter
import spacy.cli

#  spaCy NER model for biomedical entities
# MODEL_NAME = "en_ner_bc5cdr_md"
#
# try:
#     nlp = spacy.load(MODEL_NAME)
# except OSError:
#     print(f"Model '{MODEL_NAME}' not found.")
#     spacy.cli.download(MODEL_NAME)
#     nlp = spacy.load(MODEL_NAME)



# or check models directory:
MODEL_PATH = Path(__file__).resolve().parent.parent / "models/en_ner_bc5cdr_md-0.5.4"

try:
    nlp = spacy.load(str(MODEL_PATH))
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}")
    raise e


def extract_keywords_and_entities(text):
    doc = nlp(text)

    diseases = [ent.text for ent in doc.ents if ent.label_ in ['DISEASE', 'MEDICAL_CONDITION']]
    symptoms = [ent.text for ent in doc.ents if ent.label_ == 'SYMPTOM']
    procedures = [ent.text for ent in doc.ents if ent.label_ == 'PRODUCT']

    medical_entities = set(diseases + symptoms + procedures)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    potential_keywords = nouns + noun_phrases
    keywords = [word for word in potential_keywords if word not in medical_entities]

    keyword_counts = Counter(keywords)
    sorted_keywords = [item[0] for item in keyword_counts.most_common(10)]

    return sorted_keywords, diseases, symptoms, procedures

def categorize_age(age_text):
    age_text = re.sub(r'\D', '', age_text)
    try:
        age = int(age_text)
    except ValueError:
        return None

    if age <= 2:
        return 'baby'
    elif age <= 4:
        return 'toddler'
    elif age <= 12:
        return 'child'
    elif age <= 19:
        return 'teen'
    elif age <= 64:
        return 'adult'
    elif age >= 65:
        return 'elderly'
    return None

def extract_age_gender(text):
    gender = None
    for word in text.split():
        w = word.lower()
        if w in ['he', 'man', 'male']:
            gender = 'male'
        elif w in ['she', 'woman', 'female']:
            gender = 'female'
        elif w in ['child', 'boy', 'girl']:
            gender = 'child'

    if 'elderly' in text.lower():
        return ['elderly'], gender
    if 'baby' in text.lower():
        return ['baby'], gender

    if gender:
        age_pattern = r'(\d{1,2}\s*(?:[-to]\s*\d{1,2})?\s*(years?\s*old|yrs?|yr|yo|age)?|\bin\s*his\s*\d{1,2}s?|\d{1,2}\s*[-to]\s*\d{1,2}\syears?)'
        age_matches = re.findall(age_pattern, text, flags=re.IGNORECASE)
        ages = [match[0] for match in age_matches] if age_matches else []
        categorized_ages = [categorize_age(age) for age in ages if categorize_age(age)]
        return categorized_ages, gender

    return [], None
