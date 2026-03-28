"""Fetch subtitles from online providers."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..context import FetchedSubtitle, PipelineContext
from ..logging import logger
from ..subtitle_fetch import SubtitleFetcher, SubtitleValidator
from ..subtitle_fetch.providers.animesub import AnimeSubProvider
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

        try:
            all_matches = fetcher.search_all(ctx.identity, ['eng', 'pol'])
        except Exception as e:
            logger.warning(f'Subtitle search failed: {e}')
            ctx.fetched_subtitles = {}
            return ctx

        if not all_matches:
            logger.info('No subtitles found from any provider')
            ctx.fetched_subtitles = {}
            return ctx

        logger.info(f'Found {len(all_matches)} subtitle candidate(s)')

        # Download all candidates
        candidates_dir = ctx.work_dir / 'candidates'
        candidates_dir.mkdir(parents=True, exist_ok=True)
        downloaded = self._download_all(fetcher, all_matches, candidates_dir)

        if not downloaded:
            logger.warning('All candidate downloads failed')
            ctx.fetched_subtitles = {}
            return ctx

        # Validate and select best per language
        ctx.fetched_subtitles = self._validate_and_select(
            downloaded,
            ctx.reference_path,
        )
        return ctx

    def _build_fetcher(self, video_path):
        providers = [AnimeSubProvider()]
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

    def _validate_and_select(self, downloaded, reference_path):
        result = {}

        if reference_path is not None:
            try:
                validator = SubtitleValidator(reference_path)
                validated = validator.validate_candidates(downloaded, min_threshold=0.3)
            except Exception as e:
                logger.warning(f'Validation failed, falling back to provider scoring: {e}')
                validated = None
        else:
            validated = None

        if validated is not None:
            if validated:
                logger.info(f'{len(validated)} candidate(s) passed validation')
                for match, path, score in validated:
                    if match.language not in result:
                        result[match.language] = FetchedSubtitle(path=path, source=match.source)
                        logger.info(
                            f'Best {match.language}: {match.release_name} '
                            f'(score: {score:.3f}, source: {match.source})'
                        )
            else:
                logger.warning('No candidates passed validation threshold')
        else:
            for match, path in downloaded:
                if match.language not in result:
                    result[match.language] = FetchedSubtitle(path=path, source=match.source)
                    logger.info(
                        f'Best {match.language} (unvalidated): {match.release_name} '
                        f'(source: {match.source})'
                    )

        return result
