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

    A candidate is dropped if its lowercase form occurs anywhere in the same
    corpus — proper nouns are consistently capitalised, English words flip
    between cases. This catches false positives like "You", "Alright", "Stop"
    that would otherwise leak untranslated through placeholder protection.
    """
    # Track lowercase forms anywhere in the corpus — if a token appears
    # lowercase, it's almost certainly an English word, not a proper noun.
    seen_lowercase: set[str] = set()
    for text in dialogue_texts:
        for tok in re.findall(r"\b[a-zA-Z']+\b", text):
            if tok == tok.lower():
                seen_lowercase.add(tok)

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
    # but are NOT proper nouns. The lowercase-occurrence filter below catches
    # most of these; this list backs it up for words that only ever appear
    # capitalised (sentence start, single-word exclamation, etc.).
    common_false_positives = {
        # Articles, conjunctions, pronouns
        'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Which',
        'Who', 'How', 'Why', 'But', 'And', 'Yet', 'For', 'Nor', 'Not', 'Now',
        'Yes', 'Yeah', 'You', 'Your', 'Their', 'Such', 'Some', 'Each', 'Every',
        # Greetings + interjections
        'Hey', 'Hi', 'Hello', 'Heya', 'Oh', 'Ah', 'Eh', 'Huh', 'Wow', 'Whoa',
        'Yo', 'Goodbye', 'Bye', 'Welcome',
        # Discourse markers
        'Well', 'So', 'Also', 'Anyway', 'Anyhow', 'Indeed', 'However', 'Besides',
        'Maybe', 'Perhaps', 'Supposedly', 'Obviously', 'Apparently', 'Honestly',
        'Right', 'Wrong', 'True', 'False', 'Fine', 'Sure',
        # Commands / direct address
        'Look', 'Come', 'Go', 'Stop', 'Wait', 'Listen', 'Hear', 'See', 'Run',
        'Hurry', 'Move', 'Stay', 'Stand', 'Sit', 'Sleep', 'Help', 'Quiet',
        'Silence', 'Enough', 'Begin', 'Start', 'Finish', 'Continue', 'Return',
        'Forward', 'Back', 'Onward', 'Charge', 'Fire', 'Attack', 'Defend',
        # Adjectives
        'Big', 'Small', 'Old', 'Young', 'New', 'Good', 'Bad', 'Great', 'Little',
        'Strong', 'Weak', 'Brave', 'Quick', 'Slow', 'Long', 'Short', 'Best',
        'Worst', 'Hard', 'Easy', 'Tough', 'Cool', 'Hot', 'Cold', 'Crazy',
        # Adverbs / qualifiers
        'Never', 'Always', 'Often', 'Sometimes', 'Quickly', 'Slowly', 'Finally',
        'Suddenly', 'Already', 'Just', 'Even', 'Still', 'Only', 'Almost',
        'Alright', 'Okay', 'Damn', 'Hell', 'Heaven',
        # Politeness / fillers
        'Please', 'Sorry', 'Pardon', 'Excuse', 'Thanks', 'Thank',
        # Common nouns sometimes capitalised at line start
        'God', 'Idiot', 'Fool', 'Bastard', 'Stupid', 'Loser', 'Coward',
        'Women', 'Men', 'Boy', 'Girl', 'Man', 'Woman', 'Boys', 'Girls',
        'Brother', 'Sister', 'Bro', 'Sis', 'Mother', 'Father', 'Mom', 'Dad',
        'Friend', 'Friends', 'Family', 'Enemy', 'Enemies',
        'Era', 'Past', 'Future', 'Branch', 'Empire', 'Pirate', 'Pirates',
        'Ship', 'Sea', 'Ocean', 'Sky', 'Day', 'Night', 'World',
        # Verbs that frequently start exclamations
        'Get', 'Let', 'Make', 'Take', 'Give', 'Bring', 'Tell', 'Show',
        'Try', 'Keep', 'Open', 'Close', 'Find', 'Lose', 'Win', 'Eat',
        'Drink', 'Speak', 'Talk', 'Shout', 'Yell', 'Scream', 'Cry',
        'Laugh', 'Smile', 'Fight', 'Kill', 'Die', 'Live', 'Love', 'Hate',
        'Believe', 'Trust', 'Forget', 'Remember', 'Understood', 'Gathered',
        'Canceled', 'Cancelled', 'Being',
    }

    # Also filter out honorifics themselves
    common_false_positives.update(h.capitalize() for h in honorifics)

    names = set()
    for word, count in cap_word_counts.items():
        if count < 3:
            continue
        if word in common_false_positives:
            continue
        # If we've also seen the word in lowercase, it's English, not a name.
        if word.lower() in seen_lowercase:
            continue
        names.add(word)

    if names:
        logger.info(f'Detected proper nouns for translation protection: {sorted(names)}')

    return names
