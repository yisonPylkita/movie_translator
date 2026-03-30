import re
from dataclasses import dataclass

PHRASE_BASE_MAP = {
    'yes': 'tak',
    'no': 'nie',
    'maybe': 'może',
    'wait': 'czekaj',
    'stop': 'stop',
    'go': 'idź',
    'help': 'pomocy',
    'please': 'proszę',
    'thanks': 'dzięki',
    'sorry': 'przepraszam',
    'okay': 'dobrze',
    'ok': 'ok',
    'fine': 'dobrze',
    'sure': 'jasne',
    'never': 'nigdy',
    'always': 'zawsze',
    'hello': 'cześć',
    'hi': 'cześć',
    'bye': 'pa',
    'goodbye': 'do widzenia',
}

MULTI_WORD_PHRASES = {
    'thank you': 'dziękuję',
    'i see': 'rozumiem',
    'i know': 'wiem',
    'of course': 'oczywiście',
    'excuse me': 'przepraszam',
    'good luck': 'powodzenia',
}

IDIOM_PATTERNS = [
    (re.compile(r'\bbreak a leg\b', re.IGNORECASE), 'good luck'),
    (re.compile(r'\braining cats and dogs\b', re.IGNORECASE), 'raining heavily'),
    (re.compile(r'\bpiece of cake\b', re.IGNORECASE), 'very easy'),
    (re.compile(r'\bhit the nail on the head\b', re.IGNORECASE), 'exactly right'),
    (re.compile(r'\blet the cat out of the bag\b', re.IGNORECASE), 'reveal a secret'),
    (re.compile(r'\bonce in a blue moon\b', re.IGNORECASE), 'very rarely'),
    (re.compile(r'\bunder the weather\b', re.IGNORECASE), 'feeling sick'),
    (re.compile(r'\bspill the beans\b', re.IGNORECASE), 'reveal a secret'),
    (re.compile(r'\bbarking up the wrong tree\b', re.IGNORECASE), 'looking in the wrong place'),
    (re.compile(r'\bcost an arm and a leg\b', re.IGNORECASE), 'very expensive'),
]

# Patterns for content that should pass through translation untouched.
# Each tuple: (compiled regex, group name for the placeholder tag).
_PLACEHOLDER_PATTERNS = [
    (re.compile(r'\b\d{1,3}([-.)\s]\d{2,4}){2,}\b'), 'PHONE'),  # phone numbers
    (re.compile(r'\b\d{1,2}[/:]\d{2}(?:[/:]\d{2,4})?\b'), 'TIME'),  # 12:30, 1/01/2025
    (re.compile(r'https?://\S+'), 'URL'),  # URLs
    (re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'), 'NAME'),  # Title Case names
]


@dataclass
class PreprocessingStats:
    single_word_hits: int = 0
    multi_word_hits: int = 0
    idiom_hits: int = 0
    placeholder_hits: int = 0
    total_processed: int = 0

    def record_single_word(self):
        self.single_word_hits += 1
        self.total_processed += 1

    def record_multi_word(self):
        self.multi_word_hits += 1
        self.total_processed += 1

    def record_idiom(self):
        self.idiom_hits += 1

    def record_placeholder(self):
        self.placeholder_hits += 1

    def record_processed(self):
        self.total_processed += 1

    def get_summary(self) -> str:
        if self.total_processed == 0:
            return 'No preprocessing performed'

        total_hits = self.single_word_hits + self.multi_word_hits
        hit_rate = (total_hits / self.total_processed * 100) if self.total_processed > 0 else 0

        lines = [
            'Preprocessing Statistics:',
            f'  Total lines processed: {self.total_processed}',
            f'  Single-word matches: {self.single_word_hits}',
            f'  Multi-word matches: {self.multi_word_hits}',
            f'  Idiom replacements: {self.idiom_hits}',
            f'  Placeholder protections: {self.placeholder_hits}',
            f'  Direct translation rate: {hit_rate:.1f}% (skipped model)',
        ]
        return '\n'.join(lines)

    def reset(self):
        self.single_word_hits = 0
        self.multi_word_hits = 0
        self.idiom_hits = 0
        self.placeholder_hits = 0
        self.total_processed = 0


def normalize_for_lookup(text: str) -> tuple[str, str, str]:
    stripped = text.strip()

    match = re.match(r'^(.*?)([.!?,;:…]+)?$', stripped)
    base = match.group(1).strip() if match else stripped
    punct = match.group(2) or '' if match else ''

    if base.isupper():
        cap_pattern = 'UPPER'
    elif base and base[0].isupper():
        cap_pattern = 'Title'
    else:
        cap_pattern = 'lower'

    return base.lower(), punct, cap_pattern


def apply_formatting(translated: str, punct: str, cap_pattern: str) -> str:
    result = translated

    if cap_pattern == 'UPPER':
        result = result.upper()
    elif cap_pattern == 'Title' and result:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

    return result + punct


def preprocess_for_translation(
    text: str, stats: PreprocessingStats | None = None
) -> tuple[str, bool]:
    base, punct, cap = normalize_for_lookup(text)

    if base in PHRASE_BASE_MAP:
        if stats:
            stats.record_single_word()
        translated = PHRASE_BASE_MAP[base]
        formatted = apply_formatting(translated, punct, cap)
        return formatted, True

    if base in MULTI_WORD_PHRASES:
        if stats:
            stats.record_multi_word()
        translated = MULTI_WORD_PHRASES[base]
        formatted = apply_formatting(translated, punct, cap)
        return formatted, True

    processed = text
    idiom_matched = False
    for compiled_pattern, replacement in IDIOM_PATTERNS:
        new_processed = compiled_pattern.sub(replacement, processed)
        if new_processed != processed:
            idiom_matched = True
            processed = new_processed

    if stats:
        if idiom_matched:
            stats.record_idiom()
        stats.record_processed()

    return processed, False


def postprocess_translation(text: str) -> str:
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
    pattern = r'^(\w+),\s*\1([.!?])$'
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1) + match.group(2)

    return text


def _normalize_punctuation(text: str) -> str:
    text = re.sub(r'([.!?])\1+', r'\1', text)

    text = re.sub(r'\s+([.!?])', r'\1', text)

    return text


# ---------------------------------------------------------------------------
# Placeholder protection for untranslatable content
# ---------------------------------------------------------------------------


def extract_placeholders(
    text: str, stats: PreprocessingStats | None = None
) -> tuple[str, dict[str, str]]:
    """Replace numbers, URLs, and Title Case names with placeholders.

    Returns the modified text and a mapping from placeholder tags back to
    original values so they can be restored after translation.
    """
    mapping: dict[str, str] = {}
    result = text
    counter = 0

    for pattern, tag in _PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(result):
            key = f'__{tag}{counter}__'
            mapping[key] = match.group()
            counter += 1
            if stats:
                stats.record_placeholder()

    # Replace longest matches first to avoid partial overlap issues.
    for key, original in sorted(mapping.items(), key=lambda kv: -len(kv[1])):
        result = result.replace(original, key, 1)

    return result, mapping


def restore_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Restore placeholder tags with their original values."""
    result = text
    for key, original in mapping.items():
        result = result.replace(key, original)
    return result
