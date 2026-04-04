"""Proper noun detection for translation protection.

Builds a set of character names and proper nouns that should not be
translated, based on media identity and subtitle content analysis.
"""

import re

from ..logging import logger


def extract_proper_nouns_from_subtitles(dialogue_texts: list[str]) -> set[str]:
    """Extract likely proper nouns from English subtitle text.

    Uses heuristics to identify character names:
    - Words that appear in direct address patterns (e.g. "Guts!", "Sir Griffith,")
    - Capitalized single words that appear 3+ times and aren't common English words
    - Title Case words preceded by honorifics (Sir, Lord, Lady, Princess, etc.)
    """
    # Count capitalized words (not at sentence start)
    cap_word_counts: dict[str, int] = {}
    honorifics = {'sir', 'lord', 'lady', 'princess', 'prince', 'king', 'queen', 'master', 'miss'}

    for text in dialogue_texts:
        # Find words after honorifics
        for match in re.finditer(
            r'\b(?:Sir|Lord|Lady|Princess|Prince|King|Queen|Master|Miss)\s+([A-Z][a-z]+)',
            text,
        ):
            name = match.group(1)
            cap_word_counts[name] = cap_word_counts.get(name, 0) + 5  # boost honorific names

        # Find capitalized words in direct address (followed by comma, ! or ?)
        for match in re.finditer(r'\b([A-Z][a-z]{2,})[,!?]', text):
            word = match.group(1)
            cap_word_counts[word] = cap_word_counts.get(word, 0) + 2

        # Find standalone exclamations (whole line is just a name)
        stripped = text.strip().rstrip('!?.').strip()
        if re.match(r'^[A-Z][a-z]+$', stripped):
            cap_word_counts[stripped] = cap_word_counts.get(stripped, 0) + 3

        # Count all mid-sentence capitalized words
        words = text.split()
        for i, word in enumerate(words):
            if i == 0:
                continue  # skip sentence start
            clean = word.strip('.,!?;:"\'-')
            if re.match(r'^[A-Z][a-z]{2,}$', clean):
                cap_word_counts[clean] = cap_word_counts.get(clean, 0) + 1

    # Common English words that are capitalized mid-sentence in subtitles
    # but are NOT proper nouns
    common_false_positives = {
        'The',
        'This',
        'That',
        'What',
        'When',
        'Where',
        'Which',
        'Who',
        'How',
        'Why',
        'But',
        'And',
        'Yet',
        'For',
        'Nor',
        'Not',
        'Now',
        'Yes',
        'Hey',
        'Well',
        'Look',
        'Come',
        'Here',
        'There',
        'Big',
        'Old',
        'New',
        'Good',
        'Bad',
        'Great',
        'Little',
        'Never',
        'Always',
        'Perhaps',
        'Maybe',
        'Please',
        'Sorry',
        'Damn',
        'Hell',
        'God',
        'Idiot',
        'Fool',
        'Bastard',
        'Women',
        'Men',
        'Indeed',
        'However',
        'Besides',
        'Supposedly',
        'Being',
        'Those',
        'Such',
        'Some',
        'Get',
        'Let',
    }

    # Also filter out honorifics themselves
    common_false_positives.update(h.capitalize() for h in honorifics)

    names = set()
    for word, count in cap_word_counts.items():
        if count >= 3 and word not in common_false_positives:
            names.add(word)

    if names:
        logger.info(f'Detected proper nouns for translation protection: {sorted(names)}')

    return names
