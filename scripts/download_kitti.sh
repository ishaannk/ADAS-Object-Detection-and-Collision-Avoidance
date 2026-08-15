#!/usr/bin/env bash
# Downloads the KITTI Object Detection Benchmark only (training split).
#
# The Tracking Benchmark (sequences + oxts ego-motion, needed for real
# multi-frame time-to-collision) is intentionally skipped here — it adds
# ~94GB and this workspace's disk quota can't absorb it. See docs/PLAN.md.
#
# Each zip ships a "testing" split with no ground-truth labels (held out for
# KITTI's submission server) that is roughly the same size as "training".
# We only ever extract "training/*" to avoid paying for data we can't use.
set -uo pipefail

BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
ROOT="/workspace/ishank-damageai/training/data/raw"
OBJ_DIR="$ROOT/object"
mkdir -p "$OBJ_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# Small zips (label_2, calib) — cheap enough to keep both splits.
fetch_full() {
  local url="$1" out="$2"
  local name; name="$(basename "$out")"
  if [ -f "$out.extracted" ]; then log "SKIP $name (already extracted)"; return 0; fi
  log "DOWNLOAD START $name"
  if ! curl -f -sS -C - --retry 8 --retry-delay 10 -o "$out" "$url"; then
    log "DOWNLOAD FAILED $name"; return 1
  fi
  log "DOWNLOAD DONE $name ($(du -h "$out" | cut -f1))"
  log "EXTRACT START $name"
  if ! unzip -q -o "$out" -d "$OBJ_DIR"; then log "EXTRACT FAILED $name"; return 1; fi
  touch "$out.extracted"
  log "EXTRACT DONE $name"
}

# Large zips (image_2, velodyne) — download full zip (S3 doesn't support
# partial zip extraction over HTTP), extract ONLY training/*, then delete
# the zip immediately to avoid holding both copies on a tight quota.
fetch_training_only() {
  local url="$1" out="$2"
  local name; name="$(basename "$out")"
  if [ -f "$out.extracted" ]; then log "SKIP $name (already extracted)"; return 0; fi
  log "DOWNLOAD START $name"
  if ! curl -f -sS -C - --retry 8 --retry-delay 10 -o "$out" "$url"; then
    log "DOWNLOAD FAILED $name"; return 1
  fi
  log "DOWNLOAD DONE $name ($(du -h "$out" | cut -f1))"
  log "EXTRACT START $name (training/ only)"
  if ! unzip -q -o "$out" "training/*" -d "$OBJ_DIR"; then log "EXTRACT FAILED $name"; return 1; fi
  touch "$out.extracted"
  rm -f "$out"
  log "EXTRACT DONE $name, zip removed to save space"
}

fetch_full "$BASE/data_object_label_2.zip" "$OBJ_DIR/data_object_label_2.zip"
fetch_full "$BASE/data_object_calib.zip"   "$OBJ_DIR/data_object_calib.zip"
fetch_training_only "$BASE/data_object_image_2.zip"    "$OBJ_DIR/data_object_image_2.zip"
if [ "${SKIP_VELODYNE:-0}" != "1" ]; then
  fetch_training_only "$BASE/data_object_velodyne.zip"   "$OBJ_DIR/data_object_velodyne.zip"
fi

touch "$ROOT/.object_detection_ready"
log "OBJECT DETECTION BENCHMARK READY (training split: images+labels+calib+velodyne)"
touch "$ROOT/.download_complete"
