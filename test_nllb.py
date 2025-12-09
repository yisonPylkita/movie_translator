#!/usr/bin/env python3
"""
Test NLLB-200-3.3B model for English-Polish translation
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def test_nllb_model():
    """Test NLLB-200-3.3B model for English-Polish translation."""
    model_name = 'facebook/nllb-200-3.3B'

    print('🧪 Testing NLLB-200-3.3B model...')
    print(f'📝 Model: {model_name}')

    try:
        print('1. Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('✅ Tokenizer loaded successfully')

        print('2. Loading model...')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print('✅ Model loaded successfully')

        print('3. Testing translation with NLLB language codes...')
        test_inputs = [
            'Hello, world!',
            'How are you today?',
            'Thank you very much',
            'Good morning',
        ]

        # NLLB uses specific language codes
        # English: eng_Latn
        # Polish: pol_Latn

        for test_text in test_inputs:
            print(f'\n   📝 Input: {test_text}')

            inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)

            # Generate with forced target language token
            forced_bos_token_id = tokenizer.lang_code_to_id['pol_Latn']
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=50,
                num_beams=4,
                early_stopping=True,
            )

            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f'   🗣️  Output: {translated}')

        print('\n4. Testing with movie dialogue style...')
        movie_dialogues = [
            'We need to get out of here!',
            "What's our next move?",
            'I have a bad feeling about this.',
            'Are you ready for this?',
        ]

        for dialogue in movie_dialogues:
            print(f'\n   🎬 Dialogue: {dialogue}')

            inputs = tokenizer(dialogue, return_tensors='pt', padding=True, truncation=True)
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id['pol_Latn'],
                max_length=50,
                num_beams=4,
                early_stopping=True,
            )

            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f'   🎭 Polish: {translated}')

        return True

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback

        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_nllb_model()

    if success:
        print('\n🎉 SUCCESS! NLLB-200-3.3B is working!')
        print('📋 This model can now be added to the translation system.')
    else:
        print('\n❌ NLLB-200-3.3B not working.')
