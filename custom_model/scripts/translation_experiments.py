#!/usr/bin/env python3
"""Translation quality experiments for Allegro BiDi and Apple Translation backends.

Tests multiple approaches to improve translation quality on Berserk S01E11.
Each experiment translates the same set of English lines and outputs results
for comparison.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENGLISH_LINES_PATH = Path('/tmp/berserk_compare/english_lines.txt')

# Known character/proper noun list for Berserk
BERSERK_NAMES = {
    'Guts',
    'Griffith',
    'Casca',
    'Judeau',
    'Pippin',
    'Rickert',
    'Corkus',
    'Charlotte',
    'Adon',
    'Coborlwitz',
    'Ganzansenpu',
    'Band of the Hawk',
    'Hawks',
    'Raiders',
    'Blue Whale Super Strong Heavy Assault Knights',
}

# Common anime/medieval terms that should be translated specifically
GLOSSARY = {
    'march': 'Naprzód',  # military command, NOT the month
    'charge': 'Do ataku',  # military, NOT battery/money
    'way to go': 'Tak trzymaj',
    'big sister': 'Siostro',  # anime context - respectful address
    'get it together': 'Weź się w garść',
    'son of a bitch': 'Sukinsyn',
    'company commander': 'dowódca kompanii',
    'troop commander': 'dowódca oddziału',
}


def load_english_lines() -> list[str]:
    return [line.strip() for line in ENGLISH_LINES_PATH.read_text().splitlines() if line.strip()]


def protect_names(text: str, names: set[str]) -> tuple[str, dict[str, str]]:
    """Replace known proper nouns with placeholders before translation."""
    mapping = {}
    result = text
    counter = 0
    # Sort by length (longest first) to avoid partial replacements
    for name in sorted(names, key=len, reverse=True):
        pattern = re.compile(re.escape(str(name)), re.IGNORECASE)
        for match in pattern.finditer(result):
            key = f'__PN{counter}__'
            original = match.group()
            mapping[key] = original
            counter += 1
    # Replace longest first
    for key, original in sorted(mapping.items(), key=lambda kv: -len(kv[1])):
        result = result.replace(original, key, 1)
    return result, mapping


def restore_names(text: str, mapping: dict[str, str]) -> str:
    for key, original in mapping.items():
        text = text.replace(key, original)
    return text


def apply_glossary_pre(text: str) -> str:
    """Pre-translation: replace known problematic phrases."""
    lower = text.lower().strip().rstrip('!')
    # Exact match for single-word commands
    if lower == 'march':
        return 'Naprzód!'
    if lower == 'charge':
        return 'Do ataku!'
    return text


def apply_glossary_post(text: str) -> str:
    """Post-translation: fix known bad translations."""
    # Fix "Marzec" (month March) -> "Naprzód" (military March)
    text = re.sub(r'\bMarzec\b', 'Naprzód', text)
    # Fix "Dobra droga" -> "Tak trzymaj"
    text = re.sub(r'\bDobra droga\b', 'Tak trzymaj', text, flags=re.IGNORECASE)
    # Fix "Wnętrzności" -> "Guts" (character name)
    text = re.sub(r'\bWnętrzności\b', 'Guts', text, flags=re.IGNORECASE)
    # Fix "Kaska" -> "Casca"
    text = re.sub(r'\bKaska\b', 'Casca', text, flags=re.IGNORECASE)
    # Fix "strajk" (labour strike) -> "cios" (hit/strike)
    text = re.sub(r'\bstrajk\b', 'cios', text, flags=re.IGNORECASE)
    # Fix "Ładuj" (load/charge battery) -> "Do ataku" (charge!)
    text = re.sub(r'^Ładuj!$', 'Do ataku!', text)
    # Fix "Weź to razem" -> "Weź się w garść"
    text = re.sub(r'Weź to razem', 'Weź się w garść', text, flags=re.IGNORECASE)
    return text


# ── Experiment runners ──────────────────────────────────────────────────────


def run_allegro_baseline(texts: list[str]) -> list[str]:
    """Baseline Allegro: current defaults (batch_size=16, num_beams=1)."""
    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    merged, groups = merge_for_translation(texts)
    translated = translator.translate_texts(merged)
    return unmerge_translations(translated, groups, texts)


def run_allegro_beams4(texts: list[str]) -> list[str]:
    """Allegro with beam search (num_beams=4)."""
    import torch

    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    # Monkey-patch to use beam search
    def beam_generate(encoded):
        assert translator.model is not None
        with torch.inference_mode():
            return translator.model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

    translator._generate_translations = beam_generate  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]

    merged, groups = merge_for_translation(texts)
    translated = translator.translate_texts(merged)
    return unmerge_translations(translated, groups, texts)


def run_allegro_beams4_lenpen(texts: list[str]) -> list[str]:
    """Allegro with beam search + length penalty to avoid truncation."""
    import torch

    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    def beam_generate(encoded):
        assert translator.model is not None
        with torch.inference_mode():
            return translator.model.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                do_sample=False,
            )

    translator._generate_translations = beam_generate  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]

    merged, groups = merge_for_translation(texts)
    translated = translator.translate_texts(merged)
    return unmerge_translations(translated, groups, texts)


def run_allegro_name_protected(texts: list[str]) -> list[str]:
    """Allegro with proper noun protection (names replaced before translation)."""
    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    # Protect names before merging
    protected_texts = []
    mappings = []
    for t in texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    merged, groups = merge_for_translation(protected_texts)
    translated = translator.translate_texts(merged)
    result = unmerge_translations(translated, groups, protected_texts)

    # Restore names
    return [restore_names(r, m) for r, m in zip(result, mappings, strict=True)]


def run_allegro_name_protected_beams4(texts: list[str]) -> list[str]:
    """Allegro with name protection + beam search."""
    import torch

    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    def beam_generate(encoded):
        assert translator.model is not None
        with torch.inference_mode():
            return translator.model.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                do_sample=False,
            )

    translator._generate_translations = beam_generate  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]

    protected_texts = []
    mappings = []
    for t in texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    merged, groups = merge_for_translation(protected_texts)
    translated = translator.translate_texts(merged)
    result = unmerge_translations(translated, groups, protected_texts)

    return [restore_names(r, m) for r, m in zip(result, mappings, strict=True)]


def run_allegro_full_pipeline(texts: list[str]) -> list[str]:
    """Allegro with name protection + beams + glossary pre/post."""
    import torch

    from movie_translator.translation.sentence_merger import (
        merge_for_translation,
        unmerge_translations,
    )
    from movie_translator.translation.translator import SubtitleTranslator

    translator = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    translator.load_model()

    def beam_generate(encoded):
        assert translator.model is not None
        with torch.inference_mode():
            return translator.model.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                do_sample=False,
            )

    translator._generate_translations = beam_generate  # type: ignore[invalid-assignment]  # ty:ignore[invalid-assignment]

    # Phase 1: glossary pre-processing (catch exact matches before they hit the model)
    glossary_results = {}
    model_texts = []
    model_indices = []
    for i, t in enumerate(texts):
        pre = apply_glossary_pre(t)
        if pre != t:
            glossary_results[i] = pre
        else:
            model_texts.append(t)
            model_indices.append(i)

    # Phase 2: name protection
    protected_texts = []
    mappings = []
    for t in model_texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    # Phase 3: translate
    merged, groups = merge_for_translation(protected_texts)
    translated = translator.translate_texts(merged)
    model_results = unmerge_translations(translated, groups, protected_texts)

    # Phase 4: restore names + glossary post-processing
    final = [''] * len(texts)
    for i, val in glossary_results.items():
        final[i] = val
    for idx, (result, mapping) in zip(
        model_indices, zip(model_results, mappings, strict=True), strict=True
    ):
        restored = restore_names(result, mapping)
        final[idx] = apply_glossary_post(restored)

    return final


def run_apple_baseline(texts: list[str]) -> list[str]:
    """Apple Translation baseline: current defaults."""
    from movie_translator.translation.apple_backend import AppleTranslationBackend

    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)
    return backend.translate_texts(texts)


def run_apple_name_protected(texts: list[str]) -> list[str]:
    """Apple with proper noun protection."""
    from movie_translator.translation.apple_backend import AppleTranslationBackend

    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)

    protected_texts = []
    mappings = []
    for t in texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    translated = backend.translate_texts(protected_texts)
    return [restore_names(r, m) for r, m in zip(translated, mappings, strict=True)]


def run_apple_full_pipeline(texts: list[str]) -> list[str]:
    """Apple with name protection + glossary pre/post."""
    from movie_translator.translation.apple_backend import AppleTranslationBackend

    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)

    glossary_results = {}
    model_texts = []
    model_indices = []
    for i, t in enumerate(texts):
        pre = apply_glossary_pre(t)
        if pre != t:
            glossary_results[i] = pre
        else:
            model_texts.append(t)
            model_indices.append(i)

    protected_texts = []
    mappings = []
    for t in model_texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    translated = backend.translate_texts(protected_texts)

    final = [''] * len(texts)
    for i, val in glossary_results.items():
        final[i] = val
    for idx, (result, mapping) in zip(
        model_indices, zip(translated, mappings, strict=True), strict=True
    ):
        restored = restore_names(result, mapping)
        final[idx] = apply_glossary_post(restored)

    return final


def run_apple_context_window(texts: list[str]) -> list[str]:
    """Apple with context window: translate 3 lines at a time with surrounding context."""
    from movie_translator.translation.apple_backend import AppleTranslationBackend

    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)

    # Build context-enriched inputs: "[prev] ||| [current] ||| [next]"
    # Ask Apple to translate the whole thing, then extract the middle part
    WINDOW = 1  # lines of context on each side
    SEP = ' ||| '

    batched_inputs = []
    for i in range(len(texts)):
        parts = []
        for j in range(max(0, i - WINDOW), min(len(texts), i + WINDOW + 1)):
            parts.append(texts[j])
        batched_inputs.append(SEP.join(parts))

    translated = backend.translate_texts(batched_inputs)

    # Extract the middle segment from each result
    results = []
    for i, t in enumerate(translated):
        parts = t.split('|||')
        start = min(i, WINDOW)
        if len(parts) > start:
            results.append(parts[start].strip())
        else:
            # Fallback: take the whole thing
            results.append(t.strip())

    return results


def run_apple_full_with_context(texts: list[str]) -> list[str]:
    """Apple with name protection + glossary + context window."""
    from movie_translator.translation.apple_backend import AppleTranslationBackend

    backend = AppleTranslationBackend(batch_size=200, enable_enhancements=False)

    # Phase 1: glossary pre-processing
    glossary_results = {}
    model_texts = []
    model_indices = []
    for i, t in enumerate(texts):
        pre = apply_glossary_pre(t)
        if pre != t:
            glossary_results[i] = pre
        else:
            model_texts.append(t)
            model_indices.append(i)

    # Phase 2: name protection
    protected_texts = []
    mappings = []
    for t in model_texts:
        protected, mapping = protect_names(t, BERSERK_NAMES)
        protected_texts.append(protected)
        mappings.append(mapping)

    # Phase 3: context window translation
    WINDOW = 1
    SEP = ' ||| '
    batched = []
    for i in range(len(protected_texts)):
        parts = []
        for j in range(max(0, i - WINDOW), min(len(protected_texts), i + WINDOW + 1)):
            parts.append(protected_texts[j])
        batched.append(SEP.join(parts))

    translated = backend.translate_texts(batched)

    model_results = []
    for i, t in enumerate(translated):
        parts = t.split('|||')
        start = min(i, WINDOW)
        if len(parts) > start:
            model_results.append(parts[start].strip())
        else:
            model_results.append(t.strip())

    # Phase 4: restore + post-process
    final = [''] * len(texts)
    for i, val in glossary_results.items():
        final[i] = val
    for idx, (result, mapping) in zip(
        model_indices, zip(model_results, mappings, strict=True), strict=True
    ):
        restored = restore_names(result, mapping)
        final[idx] = apply_glossary_post(restored)

    return final


# ── Key diagnostic lines ────────────────────────────────────────────────────

# Line indices that expose known problems (0-indexed)
DIAGNOSTIC_LINES = {
    0: 'opening monologue',
    1: 'philosophical question',
    4: 'complex grammar',
    8: 'idiom - heart and soul',
    9: 'subjunctive mood',
    16: 'lodestone technical term',
    24: 'companion stones legend',
    34: 'complex emotion - fear',
    48: 'lesser stature - archaic',
    54: 'class/social standing',
    67: 'MARCH! (military command)',
    75: 'CHARGE! (military)',
    79: 'Band of the Hawk proper noun',
    82: 'gender - mere woman as knight',
    88: "Griffith's bed - innuendo",
    96: 'Lord Adon - pompous speech',
    112: 'dressed as man - gender',
    122: 'GUTS - character name',
    124: 'GUTS! - character name excl',
    129: 'blocking my strike',
    131: 'pulverize marble',
    141: 'Way to go! - idiom',
    144: 'Get it together - idiom',
    150: 'water with armor',
    161: 'cliff/river narrative',
    175: 'skilled commanders',
    178: 'shed blood - poetic',
}


def score_line(english: str, translated: str, line_idx: int) -> dict:
    """Quick automated quality checks on a translated line."""
    issues = []

    # Check for untranslated English words (excluding proper nouns)
    eng_words = set(re.findall(r'\b[a-z]{4,}\b', translated.lower()))
    common_eng = eng_words - {n.lower() for n in BERSERK_NAMES} - {'lord', 'sir'}
    if common_eng:
        issues.append(f'untranslated: {common_eng}')

    # Check for "Wnętrzności" (mistranslated Guts)
    if 'wnętrzności' in translated.lower() or 'wnetrznosci' in translated.lower():
        issues.append('GUTS→Wnętrzności')

    # Check for "Kaska" (mangled Casca)
    if re.search(r'\bKaska\b', translated):
        issues.append('Casca→Kaska')

    # Check for "Marzec" (month instead of military March)
    if re.search(r'\bMarzec\b', translated):
        issues.append('March→Marzec')

    # Check for "Ładuj" (charge as in load)
    if re.search(r'^Ładuj!?$', translated.strip()):
        issues.append('Charge→Ładuj')

    # Check for broken grammar (random capitals mid-sentence)
    mid_caps = re.findall(r'[a-ząćęłńóśźż]\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]', translated)
    # Filter out sentence starts after periods
    real_mid_caps = [m for m in mid_caps if '.' not in m and '!' not in m and '?' not in m]
    if real_mid_caps:
        issues.append(f'mid-sentence caps: {real_mid_caps[:2]}')

    # Check for "strajk" (labour strike instead of hit/strike)
    if 'strajk' in translated.lower():
        issues.append('strike→strajk')

    return {
        'line': line_idx,
        'english': english[:60],
        'translated': translated[:80],
        'issues': issues,
        'issue_count': len(issues),
    }


# ── Main ────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ('allegro_baseline', run_allegro_baseline),
    ('allegro_beams4', run_allegro_beams4),
    ('allegro_beams4_lenpen', run_allegro_beams4_lenpen),
    ('allegro_name_protected', run_allegro_name_protected),
    ('allegro_names_beams4', run_allegro_name_protected_beams4),
    ('allegro_full', run_allegro_full_pipeline),
    ('apple_baseline', run_apple_baseline),
    ('apple_name_protected', run_apple_name_protected),
    ('apple_full', run_apple_full_pipeline),
    ('apple_context_window', run_apple_context_window),
    ('apple_full_context', run_apple_full_with_context),
]


def main():
    texts = load_english_lines()
    print(f'Loaded {len(texts)} English dialogue lines')
    print(f'Diagnostic lines: {len(DIAGNOSTIC_LINES)}')
    print()

    results = {}
    output_dir = Path('/tmp/berserk_compare/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in EXPERIMENTS:
        print(f'{"=" * 60}')
        print(f'Running: {name}')
        print(f'{"=" * 60}')

        t0 = time.perf_counter()
        try:
            translated = fn(texts)
            elapsed = time.perf_counter() - t0
            print(f'  Done in {elapsed:.1f}s')

            # Save full output
            (output_dir / f'{name}.txt').write_text('\n'.join(translated) + '\n', encoding='utf-8')

            # Score diagnostic lines
            total_issues = 0
            diag_results = []
            for idx, desc in sorted(DIAGNOSTIC_LINES.items()):
                if idx < len(translated):
                    score = score_line(texts[idx], translated[idx], idx)
                    total_issues += score['issue_count']
                    diag_results.append(score)
                    if score['issues']:
                        print(f'  [{idx:3d}] {desc}: {", ".join(score["issues"])}')

            results[name] = {
                'elapsed': elapsed,
                'total_lines': len(translated),
                'diagnostic_issues': total_issues,
                'diagnostics': diag_results,
            }

            # Show a few sample lines
            for idx in [0, 67, 122, 141]:
                if idx < len(translated):
                    print(f'  [{idx:3d}] EN: {texts[idx][:60]}')
                    print(f'        PL: {translated[idx][:60]}')

            print()

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f'  FAILED after {elapsed:.1f}s: {e}')
            results[name] = {'elapsed': elapsed, 'error': str(e)}
            print()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"Experiment":<30} {"Time":>8} {"Issues":>8}')
    print('-' * 48)
    for name in [n for n, _ in EXPERIMENTS]:
        r = results.get(name, {})
        if 'error' in r:
            print(f'{name:<30} {"FAILED":>8} {"":>8}')
        else:
            print(f'{name:<30} {r["elapsed"]:>7.1f}s {r.get("diagnostic_issues", "?"):>8}')

    # Save all results
    (output_dir / 'results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )
    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()
