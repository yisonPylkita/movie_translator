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


@dataclass
class PreprocessingStats:
    single_word_hits: int = 0
    multi_word_hits: int = 0
    idiom_hits: int = 0
    total_processed: int = 0

    def record_single_word(self):
        self.single_word_hits += 1
        self.total_processed += 1

    def record_multi_word(self):
        self.multi_word_hits += 1
        self.total_processed += 1

    def record_idiom(self):
        self.idiom_hits += 1

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
            f'  Direct translation rate: {hit_rate:.1f}% (skipped model)',
        ]
        return '\n'.join(lines)

    def reset(self):
        self.single_word_hits = 0
        self.multi_word_hits = 0
        self.idiom_hits = 0
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
    for pattern, replacement, flags in IDIOM_PATTERNS:
        new_processed = re.sub(pattern, replacement, processed, flags=flags)
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
