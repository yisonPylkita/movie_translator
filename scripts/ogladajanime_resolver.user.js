// ==UserScript==
// @name         ogladajanime player resolver
// @namespace    movie-translator.hardsub
// @version      4.0
// @description  Auto-resolve PL-hardsub player embed URLs for ogladajanime.pl. Walks every episode, resolves curated mirrors (PL subs only, CDA then Rumble preferred), downloads canonical v2 JSON (schema_version:2) for the movie-translator pipeline.
// @updateURL    https://raw.githubusercontent.com/yisonPylkita/movie_translator/main/scripts/ogladajanime_resolver.user.js
// @note         v2: emits canonical v2 JSON schema (schema_version:2, episodes[].mirrors[]), see OUTPUT block below — mirrors the Rust parser contract.
// @note         v2: HOST_PREFERENCE promotes rumble to #2 (reliable host); CDA stays #1 (PL primary).
// @note         v2: PER_GROUP raised 2 -> 4, retains more fallback mirrors per PL sub_group.
// @match        https://ogladajanime.pl/anime/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

/*
 * WHY a userscript (not Playwright/CDP): ogladajanime detects the DevTools /
 * CDP protocol and bounces to /error/20/NN. A userscript runs in the page's
 * own JS world, so it can call the global changePlayerUrl(id) the page
 * defines and read the JSON API responses, invisible to the anti-debug.
 *
 * WHY curated (not all ~28 players): a player's host is just a mirror; the
 * thing that differs in *content* is the translation group (sub_group:
 * MioroSubs / zoro / "Nieznany"). So we keep up to PER_GROUP mirrors per PL
 * sub_group, preferring CDA (cleanest for yt-dlp) then Rumble (reliable),
 * and skip English + redundant duplicate hosts. Typically 1-4 mirrors/
 * episode instead of 28.
 *
 * Multi-episode job state lives in localStorage so it survives the page
 * navigations between episodes; the script re-inits on each episode page and
 * continues where it left off, then downloads the combined canonical JSON
 * at the end.
 *
 * OUTPUT — canonical v2 JSON schema (mirrors the Rust parser contract;
 * emitted for BOTH single-episode files anime-<slug>-ep<N>.json and the
 * full-season file anime-<slug>.json):
 *
 * {
 *   "schema_version": 2,
 *   "source_page": "https://ogladajanime.pl/anime/<slug>",
 *   "resolved_at": "<ISO8601 UTC now, e.g. 2026-07-30T12:00:00Z>",
 *   "title": "<title>",
 *   "episodes": [
 *     {
 *       "episode": 1,
 *       "mirrors": [
 *         { "host": "cda", "quality": "1080p" | null,
 *           "subtitle_group": "MioroSubs" | null, "url": "https://..." }
 *       ]
 *     }
 *   ]
 * }
 *
 * Mechanism (capture-verified): get_player_list -> catalog (.data nested
 * JSON, all groups flat); change_player_url -> .data is the embed URL.
 */
(() => {
	const SCHEMA_VERSION = 2; // canonical output schema (see OUTPUT block)
	const SUB = "pl"; // hardsub target language
	const POLL_MS = 200; // how often we check whether the response arrived
	const PACE_MS = 800; // small gap between players (the resolve wait already spaces them)
	const EPISODE_GAP_MS = 300; // near-immediate hop to the next episode
	const RECLICK_MS = 2500; // re-click play if the catalog hasn't arrived yet
	const RESOLVE_TIMEOUT_MS = 12000; // MAX wait per player (we stop early on response)
	const CATALOG_TIMEOUT_MS = 15000; // MAX wait for the player list (stop early on response)
	const LS_KEY = "oga_resolver_job";

	// Lower index = more preferred. CDA first (best for yt-dlp), Rumble #2
	// (reliable host). Unknown hosts sort last.
	const HOST_PREFERENCE = [
		"cda",
		"rumble",
		"sibnet",
		"vk",
		"mega",
		"ok",
		"dood",
		"myvi",
		"google",
		"hqq",
		"voe",
		"mp4upload",
	];

	// Derive the canonical host label from a player/embed URL. Known host
	// patterns map to stable labels consumed by the Rust parser; anything
	// unrecognised becomes "generic". Wildcard TLDs (dood.*, hqq.*, myvi.*,
	// voe.*) match on the hostname prefix.
	function hostOf(raw) {
		const s = String(raw || "").toLowerCase();
		if (!s) return "generic";
		let host = s;
		let full = s;
		try {
			const u = new URL(s);
			host = u.hostname.replace(/^www\./, "");
			full = u.href;
		} catch (e) {
			/* not a full URL — fall back to raw-string matching below */
		}
		if (host === "cda.pl" || host.endsWith(".cda.pl")) return "cda";
		if (host.includes("sibnet.ru")) return "sibnet"; // video.sibnet.ru
		if (host === "vk.com" || host.endsWith(".vk.com")) return "vk"; // incl. /video_ext.php
		if (host === "vkvideo.ru" || host.endsWith(".vkvideo.ru")) return "vk";
		if (host === "rumble.com" || host.endsWith(".rumble.com")) return "rumble";
		if (/^dood\./.test(host)) return "dood";
		if (/^hqq\./.test(host)) return "hqq";
		if (host === "mega.nz" || host.endsWith(".mega.nz")) return "mega";
		if (host === "ok.ru" || host.endsWith(".ok.ru")) return "ok";
		if (/^myvi\./.test(host)) return "myvi";
		if (host === "drive.google.com" || host.endsWith(".drive.google.com"))
			return "google";
		if (/^voe\./.test(host)) return "voe";
		if (host === "mp4upload.com" || host.endsWith(".mp4upload.com"))
			return "mp4upload";
		return "generic";
	}

	// Best-effort quality label from whatever the catalog exposes (e.g.
	// "1080p", "HD"). Numeric resolutions normalise to "NNNp"; recognised
	// letter labels pass through upper-cased; otherwise null. No brittle
	// iframe/source-tag scraping — the catalog field is authoritative.
	function qualityLabel(q) {
		const s = String(q || "").trim();
		if (!s) return null;
		const m = /\b(\d{3,4})\s*p\b/i.exec(s);
		if (m) return m[1] + "p";
		if (/^(hd|fhd|uhd|4k|sd)$/i.test(s)) return s.toUpperCase();
		return null;
	}

	// --- per-page resolve state (rebuilt each page load) ---------------------
	const state = {
		catalog: [],
		resolved: {},
		defaultUrl: null,
		episodeId: null,
	};

	// --- XHR hook: capture get_player_list + change_player_url ---------------
	const RealOpen = XMLHttpRequest.prototype.open;
	const RealSend = XMLHttpRequest.prototype.send;
	XMLHttpRequest.prototype.open = function (method, url, ...rest) {
		this.__oga_url = url;
		return RealOpen.call(this, method, url, ...rest);
	};
	XMLHttpRequest.prototype.send = function (body) {
		this.__oga_body = body;
		this.addEventListener("load", () => {
			try {
				handleResponse(
					this.__oga_url || "",
					this.__oga_body || "",
					this.responseText || "",
				);
			} catch (e) {
				console.warn("[oga-resolver] response hook error", e);
			}
		});
		return RealSend.call(this, body);
	};

	const idFromBody = (b) => {
		const m = /(?:^|&)id=(\d+)/.exec(typeof b === "string" ? b : "");
		return m ? m[1] : null;
	};

	function handleResponse(url, body, text) {
		if (url.indexOf("get_player_list") !== -1) {
			let inner;
			try {
				inner = JSON.parse(JSON.parse(text).data);
			} catch (e) {
				return;
			}
			state.defaultUrl = inner.url || state.defaultUrl;
			state.episodeId = idFromBody(body) || state.episodeId;
			state.catalog = (inner.players || []).map((p) => ({
				player_id: p.id,
				host: hostOf(p.url),
				audio: p.audio,
				sub: p.sub,
				quality: qualityLabel(p.quality),
				sub_group: p.sub_group,
			}));
			log(`catalog: ${state.catalog.length} players`);
		} else if (url.indexOf("change_player_url") !== -1) {
			let data;
			try {
				data = JSON.parse(text).data;
			} catch (e) {
				return;
			}
			if (typeof data === "string" && data.indexOf("http") === 0) {
				state.resolved[idFromBody(body) || "?"] = data;
			}
		}
	}

	// --- curation: up to PER_GROUP mirrors per PL translation group ----------
	const hostRank = (e) => {
		const i = HOST_PREFERENCE.indexOf(e.host);
		return i === -1 ? HOST_PREFERENCE.length : i;
	};
	const heightOf = (e) =>
		parseInt(String(e.quality || "").replace(/\D/g, ""), 10) || 0;

	// Keep up to this many distinct hosts per translation group, so the
	// downloader has fallback mirrors when the best one is dead (e.g. a
	// removed cda upload). Raised 2 -> 4 in v2 to retain more viable mirrors.
	const PER_GROUP = 4;

	function curate(catalog) {
		const byGroup = {};
		for (const e of catalog) {
			if ((e.sub || "") !== SUB || !e.host) continue;
			const g = e.sub_group || "";
			(byGroup[g] = byGroup[g] || []).push(e);
		}
		const picks = [];
		for (const g of Object.keys(byGroup)) {
			byGroup[g].sort(
				(a, b) => hostRank(a) - hostRank(b) || heightOf(b) - heightOf(a),
			);
			// Top-N distinct hosts per group (avoid N near-identical cda dupes).
			const seenHosts = new Set();
			for (const p of byGroup[g]) {
				if (seenHosts.has(p.host)) continue;
				seenHosts.add(p.host);
				picks.push(p);
				if (seenHosts.size >= PER_GROUP) break;
			}
		}
		// Stable, CDA-first overall ordering for the output.
		picks.sort((a, b) => hostRank(a) - hostRank(b));
		return picks;
	}

	// --- helpers -------------------------------------------------------------
	const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

	// Smart wait: poll a predicate until true or timeout. Returns how long it
	// took so we can SEE whether a step really completed (and how fast). This
	// is what replaces blind sleeps for the two variable-latency steps.
	async function waitFor(pred, timeoutMs) {
		const t0 = Date.now();
		while (Date.now() - t0 < timeoutMs) {
			if (pred()) return { ok: true, ms: Date.now() - t0 };
			await sleep(POLL_MS);
		}
		return { ok: false, ms: Date.now() - t0 };
	}
	const slugOf = () =>
		(location.pathname.match(/\/anime\/([^/]+)/) || [])[1] || "anime";
	const epOf = () =>
		parseInt(
			(location.pathname.match(/\/anime\/[^/]+\/(\d+)/) || [])[1] || "1",
			10,
		);

	// Fire a real, bubbling mouse click so the site's jQuery handlers respond
	// (a bare .click() on the wrong node, or on an SVG, does nothing).
	function realClick(el) {
		if (!el) return false;
		for (const type of ["mousedown", "mouseup", "click"]) {
			el.dispatchEvent(
				new MouseEvent(type, { bubbles: true, cancelable: true, view: window }),
			);
		}
		return true;
	}

	// The big play button is an SVG/CSS circle inside #playerStartImg (NOT an
	// <img>), so we click the container and a few fallbacks until the catalog
	// request fires. Retries because the player JS may bind its handler late.
	async function clickPlay() {
		const selectors = [
			"#playerStartImg",
			"#playerStartImg svg",
			"#playerStartImg *",
			'[onclick*="startPlayer"]',
			".player-start",
			".vjs-big-play-button",
		];
		for (const sel of selectors) {
			const el = document.querySelector(sel);
			if (el) {
				realClick(el);
				log(`clicked play via ${sel}`);
				return true;
			}
		}
		log("play button not found (selectors exhausted)");
		return false;
	}

	async function ensureCatalog() {
		const t0 = Date.now();
		let lastClick = 0;
		log("starting player…");
		while (!state.catalog.length && Date.now() - t0 < CATALOG_TIMEOUT_MS) {
			// (Re)click play until the player list request lands — the handler can
			// attach after document-idle, and a single click can be missed.
			if (Date.now() - lastClick > RECLICK_MS) {
				await clickPlay();
				lastClick = Date.now();
				log("  …waiting for player list");
			}
			await sleep(POLL_MS);
		}
		if (state.catalog.length) {
			log(
				`✓ player loaded — ${state.catalog.length} players (${((Date.now() - t0) / 1000).toFixed(1)}s)`,
			);
			return true;
		}
		log(`✗ player did NOT load within ${CATALOG_TIMEOUT_MS / 1000}s`);
		return false;
	}

	// Quality ranking for dedupe: prefer the higher resolution; a labelled
	// quality beats an unlabelled one.
	const qualityRank = (q) => {
		const n = parseInt(String(q || "").replace(/\D/g, ""), 10) || 0;
		return n || (q ? 1 : 0);
	};

	// Build a canonical v2 episode entry from resolved data. Each mirror
	// carries {host, quality, subtitle_group, url}; v1's flat urls[] is gone.
	// Dedupe on (host, url): same pair kept once, higher quality wins.
	// Each curated sub_group keeps >= 1 mirror whenever its hosts differ from
	// already-kept (host, url) pairs.
	function buildCanonical(episodeNum) {
		const picks = curate(state.catalog);
		const mirrors = [];
		const seen = new Map(); // `${host}\u0000${url}` -> index in mirrors
		for (const p of picks) {
			const pid = String(p.player_id);
			const url = state.resolved[pid];
			if (!url) continue;
			const derived = hostOf(url);
			const host = derived !== "generic" ? derived : p.host || "generic";
			const key = host + "\u0000" + url;
			const existing = seen.get(key);
			if (existing !== undefined) {
				const prev = mirrors[existing];
				if (qualityRank(prev.quality) < qualityRank(p.quality)) {
					mirrors[existing] = {
						host,
						quality: p.quality,
						subtitle_group: p.sub_group || null,
						url,
					};
				}
				continue;
			}
			seen.set(key, mirrors.length);
			mirrors.push({
				host,
				quality: p.quality,
				subtitle_group: p.sub_group || null,
				url,
			});
		}
		return { episode: episodeNum, mirrors };
	}

	async function resolveEpisode() {
		if (typeof changePlayerUrl !== "function") {
			log("ERROR: changePlayerUrl() not on page — open an episode page.");
			return null;
		}
		if (!(await ensureCatalog())) {
			log("ERROR: no catalog (play did not start).");
			return null;
		}
		const picks = curate(state.catalog);
		log(
			`curated ${picks.length} PL player(s): ${picks.map((p) => `${p.host}/${p.sub_group || "–"}`).join(", ")}`,
		);
		let i = 0;
		for (const entry of picks) {
			const pid = String(entry.player_id);
			i++;
			log(
				`  ${i}/${picks.length} ${entry.host} ${entry.quality} [${entry.sub_group || "–"}]…`,
			);
			try {
				changePlayerUrl(entry.player_id);
			} catch (e) {
				log(`    changePlayerUrl threw: ${e}`);
			}
			// Smart wait: stop the instant the change_player_url response lands.
			const r = await waitFor(() => state.resolved[pid], RESOLVE_TIMEOUT_MS);
			if (r.ok)
				log(`    ✓ ${(r.ms / 1000).toFixed(1)}s — ${state.resolved[pid]}`);
			else log(`    ✗ no response in ${(r.ms / 1000).toFixed(0)}s (skipped)`);
			await sleep(PACE_MS);
		}
		return buildCanonical(epOf());
	}

	// --- v2 canonical payload -------------------------------------------------
	function v2Payload(title, episodes) {
		return {
			schema_version: SCHEMA_VERSION,
			source_page: `${location.origin}/anime/${slugOf()}`,
			resolved_at: new Date().toISOString(),
			title: title || null,
			episodes,
		};
	}

	// --- multi-episode job (persisted across navigations) --------------------
	const loadJob = () => {
		try {
			return JSON.parse(localStorage.getItem(LS_KEY) || "null");
		} catch (e) {
			return null;
		}
	};
	const saveJob = (j) => localStorage.setItem(LS_KEY, JSON.stringify(j));
	const clearJob = () => localStorage.removeItem(LS_KEY);

	function discoverEpisodes(slug) {
		// The "Odcinki: N" count the page prints is authoritative: it's the
		// number of real episodes (1..N), excluding the episode-0 "Zapowiedź"
		// trailer that the sidebar lists but we never want. Prefer it.
		const m = /Odcinki:\s*(\d+)/i.exec(document.body.innerText || "");
		if (m) return parseInt(m[1], 10);

		// Fallbacks when the count isn't on the page: episode anchors, then a
		// sidebar row count minus the ep-0 trailer row, then the current ep.
		let max = epOf();
		const re = new RegExp(
			"/anime/" + slug.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "/(\\d+)",
		);
		document.querySelectorAll("a[href]").forEach((a) => {
			const mm = re.exec(a.getAttribute("href") || "");
			if (mm) max = Math.max(max, parseInt(mm[1], 10));
		});
		const rows = document.querySelectorAll(
			"#ep_list li, #ep_list a, #episode_table tbody tr, #episode_list li",
		).length;
		if (rows) max = Math.max(max, rows - 1); // drop the ep-0 trailer row
		return max;
	}

	// Discover optional anime display name from the page.
	function discoverTitle() {
		const h1 = document.querySelector("h1");
		if (h1 && h1.textContent) {
			const t = h1.textContent.trim();
			if (t.length > 0 && t.length < 200) return t;
		}
		return null;
	}

	function finalizeJob(job) {
		const out = v2Payload(discoverTitle(), job.results);

		const blob = new Blob([JSON.stringify(out, null, 2)], {
			type: "application/json",
		});
		const a = document.createElement("a");
		a.href = URL.createObjectURL(blob);
		const filename = `anime-${job.slug}.json`;
		a.download = filename;
		a.click();
		const total = job.results.reduce((n, r) => n + r.mirrors.length, 0);
		log(
			`JOB DONE — ${job.results.length} episodes, ${total} mirrors. Downloaded ${filename} (schema v${SCHEMA_VERSION}).`,
		);
		clearJob();
	}

	async function continueJob(job) {
		const cur = epOf();
		if (location.pathname.indexOf("/error") !== -1) {
			log("hit /error during job — waiting 8s then retrying this episode…");
			await sleep(8000);
			location.href = `/anime/${job.slug}/${job.episodes[job.idx]}`;
			return;
		}
		if (cur !== job.episodes[job.idx]) {
			// Not on the expected episode yet — nudge there.
			location.href = `/anime/${job.slug}/${job.episodes[job.idx]}`;
			return;
		}
		log(`=== episode ${cur} (${job.idx + 1}/${job.episodes.length}) ===`);
		const result = await resolveEpisode();
		if (result) job.results.push(result);
		job.idx++;
		if (job.idx < job.episodes.length) {
			saveJob(job);
			log(`→ next episode in ${EPISODE_GAP_MS / 1000}s…`);
			await sleep(EPISODE_GAP_MS);
			location.href = `/anime/${job.slug}/${job.episodes[job.idx]}`;
		} else {
			finalizeJob(job);
		}
	}

	// --- UI ------------------------------------------------------------------
	const panel = document.createElement("div");
	panel.style.cssText =
		"position:fixed;top:10px;right:10px;z-index:999999;background:#111;color:#0f0;" +
		"font:12px/1.4 monospace;padding:10px;border:1px solid #0f0;border-radius:6px;" +
		"max-width:360px;max-height:70vh;overflow:auto;box-shadow:0 2px 12px #000";
	const logBox = document.createElement("div");

	function log(msg) {
		const line = document.createElement("div");
		line.textContent = msg;
		logBox.appendChild(line);
		logBox.scrollTop = logBox.scrollHeight;
		console.log("[oga-resolver]", msg);
	}

	function renderControls() {
		const slug = slugOf();
		const maxEp = discoverEpisodes(slug);
		const curEp = epOf();
		panel.textContent = ""; // clear panel (no innerHTML — XSS-safe)

		const title = document.createElement("div");
		title.textContent = "Polish subtitle finder";
		title.style.cssText = "font-weight:bold;font-size:13px;margin-bottom:2px";
		panel.appendChild(title);

		const sub = document.createElement("div");
		sub.textContent = `${slug} · ${maxEp} episodes`;
		sub.style.cssText = "opacity:.75;margin-bottom:2px";
		panel.appendChild(sub);

		const hint = document.createElement("div");
		hint.textContent =
			"Finds the best Polish-subbed video link per episode (CDA/Rumble preferred), saves canonical v2 JSON.";
		hint.style.cssText = "opacity:.55;font-size:11px;margin-bottom:8px";
		panel.appendChild(hint);

		const bigBtn = (label) => {
			const b = document.createElement("button");
			b.textContent = label;
			b.style.cssText =
				"cursor:pointer;width:100%;margin-bottom:6px;padding:9px;font:inherit;font-size:13px";
			return b;
		};

		const allBtn = bigBtn(`⏬  All ${maxEp} episodes`);
		allBtn.onclick = () => {
			const episodes = [];
			for (let n = 1; n <= maxEp; n++) episodes.push(n); // 1..max, skips ep-0 trailer
			saveJob({ slug, episodes, idx: 0, results: [] });
			log(`starting: all ${episodes.length} episodes`);
			location.href = `/anime/${slug}/${episodes[0]}`;
		};
		panel.appendChild(allBtn);

		const oneBtn = bigBtn(`▶  This episode (#${curEp})`);
		oneBtn.onclick = async () => {
			oneBtn.disabled = true;
			const r = await resolveEpisode();
			if (r) {
				const slug = slugOf();
				const out = v2Payload(discoverTitle(), [r]);
				const blob = new Blob([JSON.stringify(out, null, 2)], {
					type: "application/json",
				});
				const a = document.createElement("a");
				a.href = URL.createObjectURL(blob);
				a.download = `anime-${slug}-ep${curEp}.json`;
				a.click();
				log(`saved ${a.download} (${r.mirrors.length} mirrors)`);
			}
			oneBtn.disabled = false;
		};
		panel.appendChild(oneBtn);

		const stop = document.createElement("div");
		stop.textContent = "cancel running job";
		stop.style.cssText =
			"cursor:pointer;text-align:center;opacity:.5;font-size:11px;text-decoration:underline";
		stop.onclick = () => {
			clearJob();
			log("job cancelled.");
		};
		panel.appendChild(stop);
		panel.appendChild(logBox);
	}

	// --- entry ---------------------------------------------------------------
	document.body.appendChild(panel);
	const job = loadJob();
	if (job && job.slug === slugOf() && job.idx < job.episodes.length) {
		panel.appendChild(logBox);
		log(
			`resuming job: episode ${job.episodes[job.idx]} (${job.idx + 1}/${job.episodes.length})`,
		);
		// Brief settle so the page's own player JS has registered; ensureCatalog
		// then re-clicks play until the list lands, so this needn't be long.
		sleep(500).then(() => continueJob(job));
	} else {
		renderControls();
	}
})();
