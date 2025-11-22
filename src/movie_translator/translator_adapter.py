from pathlib import Path

from movie_translator.constants import (
	DEFAULT_MODEL,
	LANGUAGE_POLISH,
	DEFAULT_DEVICE,
	DEFAULT_BATCH_SIZE,
)


def translate_file(
	input_path: str,
	output_path: str,
	target_language: str = LANGUAGE_POLISH,
	model: str = DEFAULT_MODEL,
	device: str = DEFAULT_DEVICE,
	batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
	from movie_translator.local_llm_provider import translate_file_local

	if not Path(input_path).exists():
		raise FileNotFoundError(f"Input subtitle file not found: {input_path}")

	translate_file_local(
		input_path=input_path,
		output_path=output_path,
		device=device,
		model_name=model,
		target_language=target_language,
		batch_size=batch_size,
	)
