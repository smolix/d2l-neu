# make/universities.mk — landing-page university adoption grid (independent of the book pipeline)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# ── Landing-page university grid ───────────────────────────
# tools/universities.json is the single source of truth for the logo grid on
# the landing page, and the evidence of record lives in the repo:
#   docs/university_courses_instructors.json  (+ the .md rendering) — ledger of
#       course, instructor, snippet and source per adoption
#   tools/universities.json — per-institution url/year/course/instructor
# Add a new adoption to the ledger AND to tools/universities.json.
# The old external evidence directory (/home/smola/d2l/data/uni_evidence/)
# is no longer needed. Pipeline:
#   1. (optional, legacy) consolidate TSV from per-region .md files, only if
#      UNI_EVIDENCE_DIR exists; skipped otherwise
#   2. tools/build_universities_json.py — merges TSV with existing logo files
#   3. tools/download_logos.py            — fetches missing logos (Wikipedia API,
#                                            incremental: rerun resumes)
#   4. tools/render_logo_grid.py          — emits <a href="..."><img></a> block
#                                            into index.md between markers

UNI_EVIDENCE_DIR ?= /home/smola/d2l/data/uni_evidence

.PHONY: universities universities-rebuild
universities:
	@mkdir -p $(LOGDIR)
	@{ if [ -d "$(UNI_EVIDENCE_DIR)" ]; then \
		( cd "$(UNI_EVIDENCE_DIR)" && python3 _consolidate.py >/dev/null ) \
		&& python3 tools/build_universities_json.py; \
	else \
		echo "note: $(UNI_EVIDENCE_DIR) not found; skipping TSV consolidation." ; \
		echo "      Evidence of record: docs/university_courses_instructors.json and tools/universities.json (edit those)."; \
	fi \
	&& python3 tools/download_logos.py \
	&& python3 tools/render_logo_grid.py; \
	} 2>&1 | tee $(LOGDIR)/universities-$(TS).log

# Force re-fetch of all logos (slow; honors Wikipedia rate limits).
universities-rebuild:
	@test -d "$(UNI_EVIDENCE_DIR)" || { echo "error: universities-rebuild regenerates tools/universities.json from $(UNI_EVIDENCE_DIR), which is missing; the evidence of record is now in the repo (docs/university_courses_instructors.json, tools/universities.json). Refusing to delete it." >&2; exit 1; }
	rm -f tools/universities.json
	$(MAKE) universities
