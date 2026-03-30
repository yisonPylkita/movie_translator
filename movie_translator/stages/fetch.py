"""Fetch subtitles from online providers."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..context import FetchedSubtitle, PipelineContext
from ..logging import logger
from ..subtitle_fetch import SubtitleFetcher, SubtitleValidator, align_ilass
from ..subtitle_fetch.align import align_to_reference as align_builtin
from ..subtitle_fetch.providers.animesub import AnimeSubProvider
from ..subtitle_fetch.providers.base import SubtitleProvider
from ..subtitle_fetch.providers.napiprojekt import NapiProjektProvider
from ..subtitle_fetch.providers.opensubtitles import OpenSubtitlesProvider
from ..subtitle_fetch.providers.podnapisi import PodnapisiProvider


class FetchSubtitlesStage:
    name = 'fetch'

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.config.enable_fetch:
            return ctx

        fetcher = self._build_fetcher(ctx.video_path)
        if fetcher is None:
            return ctx

        with ctx.metrics.span('search_all') as s:
            try:
                all_matches = fetcher.search_all(ctx.identity, ['eng', 'pol'], metrics=ctx.metrics)
            except Exception as e:
                logger.warning(f'Subtitle search failed: {e}')
                ctx.fetched_subtitles = {}
                return ctx
            s.detail('candidates', len(all_matches))

        if not all_matches:
            logger.info('No subtitles found from any provider')
            ctx.fetched_subtitles = {}
            return ctx

        logger.info(f'Found {len(all_matches)} subtitle candidate(s)')

        # Download all candidates
        candidates_dir = ctx.work_dir / 'candidates'
        candidates_dir.mkdir(parents=True, exist_ok=True)
        with ctx.metrics.span('download_all') as s:
            downloaded = self._download_all(fetcher, all_matches, candidates_dir)
            s.detail('downloaded', len(downloaded))
            s.detail('failed', len(all_matches) - len(downloaded))

        if not downloaded:
            logger.warning('All candidate downloads failed')
            ctx.fetched_subtitles = {}
            return ctx

        # Validate and select per language
        with ctx.metrics.span('validate_and_select') as s:
            ctx.fetched_subtitles, best_score = self._validate_and_select(
                downloaded,
                ctx.reference_path,
            )
            passed = sum(len(v) for v in ctx.fetched_subtitles.values())
            s.detail('passed', passed)
            s.detail('rejected', len(downloaded) - passed)
            if best_score is not None:
                s.detail('best_score', round(best_score, 3))

        # Realign fetched Polish subtitles against the English reference
        if ctx.reference_path and 'pol' in ctx.fetched_subtitles:
            for sub in ctx.fetched_subtitles['pol']:
                with ctx.metrics.span('align') as s:
                    method, offset = self._align_subtitle(sub.path, ctx.reference_path)
                    s.detail('method', method)
                    if offset is not None:
                        s.detail('offset_ms', offset)

        return ctx

    def _build_fetcher(self, video_path):
        providers: list[SubtitleProvider] = [AnimeSubProvider()]
        providers.append(PodnapisiProvider())
        napi = NapiProjektProvider()
        napi.set_video_path(video_path)
        providers.append(napi)
        api_key = os.environ.get('OPENSUBTITLES_API_KEY', '')
        if api_key:
            providers.append(OpenSubtitlesProvider(api_key=api_key))
        return SubtitleFetcher(providers)

    def _download_all(self, fetcher, matches, candidates_dir):
        downloaded = []

        def _download(i_match):
            i, match = i_match
            filename = f'{match.source}_{match.language}_{i}.{match.format}'
            output_path = candidates_dir / filename
            fetcher.download_candidate(match, output_path)
            return match, output_path

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_download, (i, m)): m for i, m in enumerate(matches)}
            for future in as_completed(futures):
                try:
                    downloaded.append(future.result())
                except Exception as e:
                    match = futures[future]
                    logger.warning(f'Failed to download candidate {match.subtitle_id}: {e}')

        logger.info(f'Downloaded {len(downloaded)} candidate(s)')
        return downloaded

    @staticmethod
    def _align_subtitle(subtitle_path, reference_path) -> tuple[str, int | None]:
        """Align a subtitle file to a reference, trying ilass first.

        Returns (method, offset_ms) where offset_ms may be None for ilass.
        """
        if align_ilass.is_available():
            if align_ilass.align_to_reference(subtitle_path, reference_path):
                return 'ilass', None
            logger.info('ilass alignment failed, falling back to built-in')
        offset = align_builtin(subtitle_path, reference_path)
        return 'builtin', offset

    # Keep all Polish subs scoring at or above this threshold.
    _QUALITY_THRESHOLD = 0.8

    def _validate_and_select(self, downloaded, reference_path):
        """Returns (result_dict, best_score) where best_score may be None."""
        result: dict[str, list[FetchedSubtitle]] = {}
        best_score = None

        if reference_path is not None:
            try:
                validator = SubtitleValidator(reference_path)
                validated = validator.validate_candidates(downloaded, min_threshold=0.5)
            except Exception as e:
                logger.warning(f'Validation failed, falling back to provider scoring: {e}')
                validated = None
        else:
            validated = None

        if validated is not None:
            if validated:
                best_score = validated[0][2]
                logger.info(f'{len(validated)} candidate(s) passed validation')
                for match, path, score in validated:
                    sub = FetchedSubtitle(path=path, source=match.source)
                    lang = match.language
                    if lang not in result:
                        # First (best) candidate for this language — always keep.
                        result[lang] = [sub]
                        logger.info(
                            f'Selected {lang}: {match.release_name} '
                            f'(score: {score:.3f}, source: {match.source})'
                        )
                    elif score >= self._QUALITY_THRESHOLD:
                        # Additional high-quality candidate — keep it too.
                        result[lang].append(sub)
                        logger.info(
                            f'Also keeping {lang}: {match.release_name} '
                            f'(score: {score:.3f}, source: {match.source})'
                        )
            else:
                logger.warning('No candidates passed validation threshold')
        else:
            for match, path in downloaded:
                if match.language not in result:
                    result[match.language] = [FetchedSubtitle(path=path, source=match.source)]
                    logger.info(
                        f'Best {match.language} (unvalidated): {match.release_name} '
                        f'(source: {match.source})'
                    )

        return result, best_score
