from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

from ..logging import logger
from ..metrics.collector import MetricsCollector, NullCollector
from .providers.base import SubtitleProvider
from .types import SubtitleMatch


class SubtitleFetcher:
    """Orchestrates subtitle search across multiple providers."""

    def __init__(self, providers: Sequence[SubtitleProvider]):
        self._providers = providers

    def search_all(
        self,
        identity,
        languages: list[str],
        metrics: MetricsCollector | NullCollector | None = None,
    ) -> list[SubtitleMatch]:
        """Search all providers in parallel, return ALL plausible matches sorted by score."""
        if metrics is None:
            metrics = NullCollector()
        all_matches: list[SubtitleMatch] = []

        def _search_provider(provider):
            with metrics.span(provider.name) as s:
                matches = provider.search(identity, languages)
                s.detail('candidates', len(matches))
                return provider.name, matches

        with ThreadPoolExecutor(max_workers=len(self._providers)) as pool:
            futures = {
                pool.submit(copy_context().run, _search_provider, p): p for p in self._providers
            }
            for future in as_completed(futures):
                try:
                    name, matches = future.result()  # ty: ignore[not-iterable]
                    all_matches.extend(matches)
                    logger.debug(f'{name}: found {len(matches)} matches')
                except Exception as e:
                    provider = futures[future]
                    logger.warning(f'{provider.name} search failed: {e}')

        return sorted(all_matches, key=lambda m: (m.score, m.hash_match), reverse=True)

    def download_candidate(self, match: SubtitleMatch, output_path: Path) -> Path:
        """Download a single candidate subtitle file. Returns path written to."""
        provider = self._find_provider(match.source)
        if provider is None:
            raise ValueError(
                f"No provider registered with name '{match.source}'. "
                f'Available providers: {[p.name for p in self._providers]}'
            )
        provider.download(match, output_path)
        return output_path

    def fetch_subtitles(
        self,
        identity,
        languages: list[str],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Search all providers and download best subtitle per language.

        Returns {language_code: subtitle_file_path} for successfully downloaded subtitles.
        """
        all_matches = self.search_all(identity, languages)

        if not all_matches:
            logger.info('No subtitles found from any provider')
            return {}

        # Pick best match per language (highest score wins, hash_match breaks ties)
        best: dict[str, SubtitleMatch] = {}
        for match in all_matches:
            if match.language not in best:
                best[match.language] = match

        # Download best matches
        result: dict[str, Path] = {}
        for lang, match in best.items():
            output_path = output_dir / f'fetched_{lang}.{match.format}'
            try:
                self.download_candidate(match, output_path)
                result[lang] = output_path
                logger.info(
                    f'Fetched {lang} subtitles: {match.release_name} '
                    f'({"hash" if match.hash_match else "query"} match, {match.source})'
                )
            except Exception as e:
                logger.warning(f'Failed to download {lang} subtitle: {e}')

        return result

    def _find_provider(self, name: str):
        for p in self._providers:
            if p.name == name:
                return p
        return None
