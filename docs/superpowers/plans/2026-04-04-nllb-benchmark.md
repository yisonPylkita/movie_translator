# NLLB Model Integration & Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate NLLB-200 distilled 600M and 1.3B models as translation backends and benchmark them against Allegro BiDi and Apple Translation on quality, speed, and memory.

**Architecture:** Add NLLB model configs to the existing model registry, extend `SubtitleTranslator` to handle NLLB's language token mechanism (`forced_bos_token_id` + `src_lang`), then run a standalone benchmark script that translates the same subtitle data through all backends and produces a comparison report.

**Tech Stack:** HuggingFace transformers (already a dependency), `facebook/nllb-200-distilled-600M`, `facebook/nllb-200-distilled-1.3B`

---

### Task 1: Add NLLB model configs to the registry

**Files:**
- Modify: `movie_translator/translation/models.py`

- [ ] **Step 1: Add NLLB entries to TRANSLATION_MODELS**

```python
TRANSLATION_MODELS: dict[str, ModelConfig] = {
    'allegro': {
        'huggingface_id': 'allegro/BiDi-eng-pol',
        'description': 'Allegro BiDi English-Polish',
        'max_length': 512,
    },
    'nllb-600m': {
        'huggingface_id': 'facebook/nllb-200-distilled-600M',
        'description': 'NLLB-200 Distilled 600M (200 languages)',
        'max_length': 512,
    },
    'nllb-1.3b': {
        'huggingface_id': 'facebook/nllb-200-distilled-1.3B',
        'description': 'NLLB-200 Distilled 1.3B (200 languages)',
        'max_length': 512,
    },
}
```

- [ ] **Step 2: Commit**

```bash
git add movie_translator/translation/models.py
git commit -m "feat: add NLLB 600M and 1.3B to model registry"
```

---

### Task 2: Extend SubtitleTranslator for NLLB language tokens

**Files:**
- Modify: `movie_translator/translation/translator.py`
- Test: `movie_translator/translation/tests/test_translator.py`

NLLB models require:
1. `tokenizer.src_lang = "eng_Latn"` set before encoding
2. `forced_bos_token_id = tokenizer.convert_tokens_to_ids("pol_Latn")` passed to `generate()`
3. No `>>pol<<` prefix (that's BiDi-only)

- [ ] **Step 1: Write failing test for NLLB preprocessing**

```python
def test_preprocess_texts_nllb_no_prefix():
    """NLLB models should NOT get the >>pol<< prefix."""
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model_config = {'huggingface_id': 'facebook/nllb-200-distilled-600M'}
    result = translator._preprocess_texts(['Hello world'])
    assert result == ['Hello world']
    assert '>>pol<<' not in result[0]
```

- [ ] **Step 2: Run test to verify it passes** (it should already pass since the BiDi check is `if 'bidi' in huggingface_id`)

Run: `pytest movie_translator/translation/tests/test_translator.py::test_preprocess_texts_nllb_no_prefix -v`

- [ ] **Step 3: Add `_is_nllb` property and modify `_load_tokenizer` and `_generate_translations`**

In `translator.py`:

```python
@property
def _is_nllb(self) -> bool:
    return 'nllb' in self.model_config.get('huggingface_id', '').lower()

def _load_tokenizer(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    if self._is_nllb:
        self.tokenizer.src_lang = 'eng_Latn'

def _generate_translations(self, encoded: dict) -> torch.Tensor:
    if self.model is None:
        raise RuntimeError('Model not loaded — call load_model() first')
    generate_kwargs = dict(
        **encoded,
        max_new_tokens=128,
        num_beams=1,
        early_stopping=True,
        do_sample=False,
    )
    if self._is_nllb and self.tokenizer is not None:
        generate_kwargs['forced_bos_token_id'] = self.tokenizer.convert_tokens_to_ids('pol_Latn')
    with torch.inference_mode():
        return self.model.generate(**generate_kwargs)
```

- [ ] **Step 4: Write test for NLLB forced_bos_token_id generation**

```python
def test_generate_translations_nllb_uses_forced_bos(mocker):
    """NLLB models must pass forced_bos_token_id to generate()."""
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model_config = {'huggingface_id': 'facebook/nllb-200-distilled-600M'}
    mock_model = mocker.MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
    translator.model = mock_model
    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 256047  # pol_Latn token ID
    translator.tokenizer = mock_tokenizer

    encoded = {'input_ids': torch.tensor([[1, 2]])}
    translator._generate_translations(encoded)

    call_kwargs = mock_model.generate.call_args[1]
    assert call_kwargs['forced_bos_token_id'] == 256047
```

- [ ] **Step 5: Run tests**

Run: `pytest movie_translator/translation/tests/test_translator.py -v -k nllb`
Expected: All NLLB tests PASS

- [ ] **Step 6: Commit**

```bash
git add movie_translator/translation/translator.py movie_translator/translation/tests/test_translator.py
git commit -m "feat: add NLLB language token handling in translator"
```

---

### Task 3: Create benchmark script

**Files:**
- Create: `benchmarks/nllb_comparison.py`

A standalone script that:
1. Parses the Oppenheimer SRT into subtitle lines
2. Translates through each backend: allegro, nllb-600m, nllb-1.3b, apple
3. Measures: model load time, translation time, peak memory (via `torch.mps.current_allocated_memory()` or `psutil`)
4. Prints comparison table and side-by-side translation samples
5. Saves results to `benchmarks/results/nllb_comparison_YYYY-MM-DD.json`

- [ ] **Step 1: Create benchmarks directory**

```bash
mkdir -p benchmarks/results
```

- [ ] **Step 2: Write the benchmark script**

The script should:
- Use `pysubs2` to parse the Oppenheimer SRT
- Instantiate `SubtitleTranslator` for each model key
- Time `load_model()` and `translate_texts()` separately
- Track memory with `resource.getrusage()` before/after
- Try Apple backend if available
- Print rich table with results
- Save JSON for later comparison

- [ ] **Step 3: Run the benchmark**

```bash
python -m benchmarks.nllb_comparison
```

- [ ] **Step 4: Commit results**

```bash
git add benchmarks/
git commit -m "feat: add NLLB benchmark script and initial results"
```

---

### Task 4: Run full benchmark on anime episode data

If extracted subtitle files are available from previous pipeline runs, also benchmark on longer anime dialogue (Berserk/Dandadan). This task depends on what data is accessible without running the full video pipeline.

- [ ] **Step 1: Check for cached extracted subtitles from previous runs**
- [ ] **Step 2: If available, add them to the benchmark run**
- [ ] **Step 3: Re-run and save updated results**
- [ ] **Step 4: Commit**
