# Detection v2 benchmark review ? 2026-11-07

Recommendation: run DEMO with small_640, then FULL if the continuous videos are acceptable. This archive is BENCHMARK, not DEMO; it contains no completed DEMO MP4 required by the FULL engine.

Data integrity: COMPLETE schema v2, 20 chunks, 2,444 processed-frame records (1,222/config), including 404 raw-empty records. Checked the inner export inventory/SHA256, source hashes against actual five MKV, exact indices/PTS/segments, model lock and COMPLETE marker. No inference or source modification performed during review.

Measured remote T4 FP16 batch 8: nano_640 18.98 predict FPS; small_640 18.90. Total benchmark wall 1,152.16 s (19.2 min); decode stage 977.08 s (16.3 min); one shared full source decode, 144,110 frames. These are different denominators: predict FPS is not end-to-end throughput.

Reviewed all 30 supplied private JPEGs. Small has fewer low-confidence furniture/edge candidates and stronger doorway detections in camera 3. Camera 2 frame 13498 is a counterexample: small misses a partially occluded person beside the chairs which nano detects. Camera 4 standing people remain detected. Camera 5 crop contributes far-zone candidates; tiny people remain uncertain. No identity labels, precision/recall, or superiority claims are inferred from these stills.

Keep code, batch=8, conf=0.1, scene rules and crop. Do not add broad masks or raise global confidence to hide errors. Practical selection is small_640, not a proven universal quality winner.

Next Kaggle settings: MODE="DEMO", SELECTED_CONFIGURATION="small_640", DEVICE="cuda:0", BATCH_SIZE=8. Run the notebook cells again. It saves five continuous 30-second MP4 plus all machine records. Download /kaggle/working/shadowtrack_detection_v2/shadowtrack_detection_demo.zip. After reviewing them, set MODE="FULL" and FULL_DEMO_DIR to the printed DEMO run directory, keep the same configuration/batch.

FULL selects all 144,110 source frames. Rough extrapolation from this short benchmark is about 2.4 hours before additional overhead; reserve approximately 2?3 hours, with no guarantee. MAX_WALL_SECONDS=10000 remains a checkpoint budget, not a Kaggle quota. Resume partial output if stopped. Preserve detection records/manifests, not only MP4.

No code changed and no new pytest run was needed; the previous 127 CPU tests remain the local baseline. Review acceptance: BENCHMARK_VALIDATED / QUALITY_PARTIAL / DEMO_REQUIRED. FULL not run.
