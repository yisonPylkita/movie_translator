# FLAN-T5 Model Integration Status

## 🎯 Current Status

The `sdadas/flan-t5-base-translator-en-pl` model has been investigated for integration as an alternative to the default Allegro BiDi model.

## ⚠️ Current Issue

The FLAN-T5 model is experiencing **tokenizer compatibility issues** with the current SentencePiece library version:

```
TypeError: not a string
```

This error occurs during tokenizer initialization and is related to SentencePiece version incompatibility.

## 🔧 Implementation Structure

The infrastructure for FLAN-T5 integration has been implemented:

### ✅ Completed
- Model configuration system in `ai_translator.py`
- CLI argument `--model flan-t5` support
- Model-specific handling in batch translation
- Proper model class selection (`T5ForConditionalGeneration`)
- Model-specific text preprocessing

### 🔄 Ready When Fixed
- Tokenizer loading with fallback options
- Model-specific generation parameters
- Error handling and graceful fallbacks

## 🛠️ How to Enable FLAN-T5

Once the tokenizer issue is resolved, enabling FLAN-T5 is straightforward:

1. **Uncomment the model configuration** in `ai_translator.py`:
```python
'flan-t5': {
    'name': 'sdadas/flan-t5-base-translator-en-pl',
    'description': 'FLAN-T5 English-Polish Translator',
    'type': 'seq2seq', 
    'max_length': 512,
},
```

2. **Add to CLI choices** in `main.py`:
```python
choices=['allegro', 'flan-t5']
```

3. **Test the model**:
```bash
python translate.py --model flan-t5 ~/Downloads/test_movies
```

## 🔍 Potential Solutions

### Option 1: SentencePiece Version Fix
```bash
pip install sentencepiece==0.1.99
```

### Option 2: Alternative Tokenizer Loading
```python
# Try different tokenizer loading approaches
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
```

### Option 3: Direct T5 Tokenizer
```python
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)
```

## 📊 Expected Performance

Based on the model description, FLAN-T5 should provide:
- **High-quality translations** (FLAN-T5 is instruction-tuned)
- **Different translation style** compared to BiDi models
- **Comparable speed** with similar memory usage

## 🎯 Current Recommendation

For now, **continue using the Allegro BiDi model** which is working perfectly:

```bash
python translate.py --model allegro ~/Downloads/test_movies
```

The FLAN-T5 integration infrastructure is ready and can be activated once the tokenizer compatibility issue is resolved.

## 📝 Testing

Use the test script to verify model availability:
```bash
python test_models.py
```

This will show which models are currently working and help identify when FLAN-T5 becomes available.
