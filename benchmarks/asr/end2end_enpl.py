"""End-to-end value check for the English-dub case. Runs in the MAIN project
venv (it imports the production translator).

Takes an EN ASR transcript (a results/*.json), translates it EN->PL with the
production Allegro translator, and writes the Polish hypothesis. A separate
scoring step (eval venv) then chrF's it against the real `pl_allegro` track —
i.e. how much ASR error survives into the final Polish vs using the clean
English subtitle the pipeline normally consumes.

    .venv/bin/python benchmarks/asr/end2end_enpl.py <transcript.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    from movie_translator.translation.translator import SubtitleTranslator

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text())
    texts = [s['text'] for s in d['segments'] if s['text'].strip()]

    tr = SubtitleTranslator(model_key='allegro', device='mps', batch_size=16)
    tr.load_model()
    pl = tr.translate_texts(texts)

    out.write_text(
        json.dumps(
            {'engine': d['engine'], 'model': d['model'], 'lang': d['lang'], 'pl': pl},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f'{d["engine"]}/{d["model"]}: translated {len(texts)} EN lines -> {out.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
