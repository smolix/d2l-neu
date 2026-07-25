# make/universities.mk — landing-page university adoption grid (independent of the book pipeline)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# ── Landing-page university grid ───────────────────────────
# tools/universities.json is the single source of truth for the logo grid on
# the landing page. The pipeline:
#   1. ../data/uni_evidence/UNIVERSITIES.tsv — consolidated course-evidence rows
#      (rebuilt by ../data/uni_evidence/_consolidate.py from per-region .md files)
#   2. tools/build_universities_json.py — merges TSV with existing logo files
#   3. tools/download_logos.py            — fetches missing logos (Wikipedia API,
#                                            incremental: rerun resumes)
#   4. tools/render_logo_grid.py          — emits <a href="..."><img></a> block
#                                            into index.md between markers
#
# Add new course evidence by appending to a region .md under
# /home/smola/d2l/data/uni_evidence/ then `make universities`.

.PHONY: universities universities-rebuild
universities:
	@mkdir -p $(LOGDIR)
	@{ ( cd /home/smola/d2l/data/uni_evidence && python3 _consolidate.py >/dev/null ) \
		&& python3 tools/build_universities_json.py \
		&& python3 tools/download_logos.py \
		&& python3 tools/render_logo_grid.py; \
	} 2>&1 | tee $(LOGDIR)/universities-$(TS).log

# Force re-fetch of all logos (slow; honors Wikipedia rate limits).
universities-rebuild:
	rm -f tools/universities.json
	$(MAKE) universities
