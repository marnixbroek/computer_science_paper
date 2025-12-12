from functools import lru_cache
import re

# Brand names extracted from dataset
BRAND_NAMES = ['sigmac', 'proscan', 'hello kitty', 'sansui', 'hannspree', 'seiki', 'westinghouse', 'hisense', 'magnavox', 'craig', 'rca', 'haier', 'jvc', 'lg', 'lg electronics', 'mitsubishi', 'hp', 'sunbritetv', 'upstar', 'panasonic', 'naxa', 'affinity', 'sharp', 'sceptre', 'epson', 'samsung', 'toshiba', 'elo', 'sanyo', 'coby', 'pyle', 'vizio', 'nec', 'supersonic', 'viewsonic', 'philips', 'sony', 'optoma', 'compaq', 'tcl', 'avue', 'insignia', 'venturer', 'dynex', 'contex']

# matches words containing both letters and numbers
pattern_title = re.compile(r'\b(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]+\b')

pattern_value = re.compile(
    r"^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$"
)

# Pattern to extract model IDs with either:
# - at least 2 letter parts and 1 digit part
# - or at least 2 digit parts and 1 letter part
# - or at least 2 letters followed by at least 2 digits
# Letters and digits can be separated by hyphens
# Matches any alphanumeric sequence with optional hyphens
alphanum_seq = re.compile(r'\b[\w-]+\b', re.IGNORECASE)

# Removes trailing alphabetic characters (e.g. "20.8lbs" → "20.8")
strip_suffix = re.compile(r"[a-zA-Z]+$")

def is_model_id(s):
    # Split into consecutive digit/letter sequences
    parts = re.findall(r'[A-Za-z]+|[0-9]+', s)
    n_letters = sum(1 for p in parts if p.isalpha())
    n_digits = sum(1 for p in parts if p.isdigit())
    # Match criteria: either 2+ letters & 1+ digits OR 2+ digits & 1+ letters OR at least 2 letters followed by at least 2 digits
    return (n_letters >= 2 and n_digits >= 1) or (n_digits >= 2 and n_letters >= 1) or (len(parts) >= 2 and parts[-2].isalpha() and len(parts[-2]) >= 2 and parts[-1].isdigit() and len(parts[-1]) >= 2)

def extract_model_id(title):
    candidates = alphanum_seq.findall(title)
    model_ids = [c for c in candidates if is_model_id(c)]
    # print("Extracted from title:", title, "->", model_ids)
    if len(model_ids) == 1:
        return model_ids[0]  # Return the model ID
    elif len(model_ids) > 1:
        # return longest model ID if multiple found
        return max(model_ids, key=len)
    else:    
        return None  # Return None if not exactly one model ID found

def extractModelWordsFromTitle(title, method):
    """Extract model words from a product title using regex pattern."""
    # Combine brand names and the regex pattern to extract model words
    matches = set(m.group(0) for m in pattern_title.finditer(title.lower()))

    # Add brand names found in the title
    if method == 'MSMSP+':
        for brand in BRAND_NAMES:
            if brand.lower() in title.lower():
                matches.add(brand.lower())

    return matches

def extractModelWordsFromValue(feature, value, filter_interesting, method):
    """Extract model words from a feature value using regex pattern."""
    modelWords = set()

    interestingFeatures = ["brand", "size", "refresh", "resolution", ["type", "technology"], "weight"]

    value = value.lower().strip()

    # Try full original value (with dots, without removing non-alphanumerics except spaces)
    # Remove spaces inside value (paper does NOT allow spaces)
    value_nospace = re.sub(r'\s+', '', value)

    # Check match
    if pattern_value.match(value_nospace):
        # If letters present: strip them
        if re.search(r'[a-z]', value_nospace):
            num = ''.join(re.findall(r'\d+(\.\d+)?', value_nospace))
            # Fix regex extraction: take first group
            num = value_nospace[:value_nospace.find(re.search(r'[a-z]', value_nospace).group())]
            modelWords.add(num)
        else:
            # Pure number
            modelWords.add(value_nospace)

    if not filter_interesting or method == 'MSMP+':
        return modelWords
    else:
        for feature_key in interestingFeatures:
            if isinstance(feature_key, list):
                if any(fk in feature.lower() for fk in feature_key):
                    return modelWords
            else:
                if feature_key in feature.lower():
                    return modelWords
    
    return set()


# Cache results of extracting model words from the same string
@lru_cache(maxsize=None)
def extractModelWordsFromValueCached(feature: str, value_str: str):
    return extractModelWordsFromValue(feature, value_str, False, 'MSMP+')

def getAllModelWords(data, product_key, feature_keys):
    """Extract all model words from the product's featuresMap.
    Model words are defined as words consisting of both numeric and non-numeric tokens. 
    """
    product = data[product_key]
    features_map = product.get('featuresMap', {})

    model_words = {
        word
        for key in feature_keys
        for word in extractModelWordsFromValueCached(str(key), str(features_map.get(key, "")))
    }

    return model_words

def getMatchingModelWordsPercentage(words1, words2):
    """Calculate the percentage of matching model words between two sets of model words."""
    if not words1 or not words2:
        return 0.0
    
    matching_words = words1.intersection(words2)
    total_words = words1.union(words2)
    
    if len(total_words) == 0:
        return 0.0
    
    return len(matching_words) / len(total_words)