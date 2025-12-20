import re

SHORT_PHRASE_MAP = {
    'Yes.': 'Tak.',
    'Yes?': 'Tak?',
    'Yes!': 'Tak!',
    'No.': 'Nie.',
    'No?': 'Nie?',
    'No!': 'Nie!',
    'Maybe.': 'Może.',
    'Maybe?': 'Może?',
    'Wait.': 'Czekaj.',
    'Wait!': 'Czekaj!',
    'Stop.': 'Stop.',
    'Stop!': 'Stop!',
    'Go.': 'Idź.',
    'Go!': 'Idź!',
    'Help!': 'Pomocy!',
    'Please.': 'Proszę.',
    'Thanks.': 'Dzięki.',
    'Sorry.': 'Przepraszam.',
    'Okay.': 'Dobrze.',
    'OK.': 'OK.',
    'Fine.': 'Dobrze.',
    'Sure.': 'Jasne.',
    'Never.': 'Nigdy.',
    'Always.': 'Zawsze.',
    'Hello.': 'Cześć.',
    'Hi.': 'Cześć.',
    'Bye.': 'Pa.',
    'Goodbye.': 'Do widzenia.',
}

IDIOM_PATTERNS = [
    (r'\bbreak a leg\b', 'good luck', re.IGNORECASE),
    (r'\braining cats and dogs\b', 'raining heavily', re.IGNORECASE),
    (r'\bpiece of cake\b', 'very easy', re.IGNORECASE),
    (r'\bhit the nail on the head\b', 'exactly right', re.IGNORECASE),
    (r'\blet the cat out of the bag\b', 'reveal a secret', re.IGNORECASE),
    (r'\bonce in a blue moon\b', 'very rarely', re.IGNORECASE),
    (r'\bunder the weather\b', 'feeling sick', re.IGNORECASE),
    (r'\bspill the beans\b', 'reveal a secret', re.IGNORECASE),
    (r'\bbarking up the wrong tree\b', 'looking in the wrong place', re.IGNORECASE),
    (r'\bcost an arm and a leg\b', 'very expensive', re.IGNORECASE),
]


def preprocess_for_translation(text: str) -> tuple[str, bool]:
    """
    Preprocess text before translation.

    Returns:
        tuple[str, bool]: (processed_text, was_mapped)
            - If was_mapped is True, skip model translation
    """
    stripped = text.strip()

    if stripped in SHORT_PHRASE_MAP:
        return SHORT_PHRASE_MAP[stripped], True

    processed = text
    for pattern, replacement, flags in IDIOM_PATTERNS:
        processed = re.sub(pattern, replacement, processed, flags=flags)

    return processed, False


def postprocess_translation(text: str) -> str:
    """Clean up common translation artifacts."""
    if not text:
        return text

    if not text.strip():
        return ''

    cleaned = text.strip()

    cleaned = _remove_dialogue_markers(cleaned)
    cleaned = _remove_repetition(cleaned)
    cleaned = _normalize_punctuation(cleaned)

    return cleaned


def _remove_dialogue_markers(text: str) -> str:
    """Remove dialogue markers like '- Text! - Text!'"""
    pattern = r'^-\s*([^-]+?)\s*!\s*-\s*\1\s*!$'
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1) + '!'

    pattern = r'^-\s*([^-]+?)\s*[.?!]\s*-\s*\1\s*[.?!]$'
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1) + text[-1]

    return text


def _remove_repetition(text: str) -> str:
    """Remove simple repetitions like 'Tak, tak.'"""
    pattern = r'^(\w+),\s*\1([.!?])$'
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1) + match.group(2)

    return text


def _normalize_punctuation(text: str) -> str:
    """Normalize excessive punctuation."""
    text = re.sub(r'([.!?])\1+', r'\1', text)

    text = re.sub(r'\s+([.!?])', r'\1', text)

    return text
