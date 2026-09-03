#!/usr/bin/env bash
# Local dev server for this site.
#   ./serve.sh start     plain Jekyll dev server (auto-regenerates)
#   ./serve.sh review    Jekyll build --watch + review server with the feedback overlay
#   ./serve.sh stop      stops whichever is running
#   ./serve.sh restart   stop, wipe caches, start (plain)
#   ./serve.sh clean     stop and wipe _site/.jekyll-cache/.jekyll-metadata
#   ./serve.sh status
set -euo pipefail

PORT=4123
HOST=0.0.0.0
LOG_FILE="/tmp/jekyll_serve.log"
REVIEW_LOG="/tmp/review_server.log"
PATTERN="jekyll serve --host $HOST --port $PORT"
WATCH_PATTERN="jekyll build --watch"
REVIEW_PATTERN="tools/review-server.py"
cd "$(dirname "$0")"

wait_up() {
  for _ in $(seq 1 30); do
    sleep 0.5
    if curl -s -o /dev/null "http://localhost:$PORT/"; then return 0; fi
  done
  return 1
}

start() {
  if pgrep -f "$PATTERN" > /dev/null; then
    echo "Already running at http://localhost:$PORT/"
    exit 0
  fi
  if pgrep -f "$REVIEW_PATTERN" > /dev/null; then
    echo "Review server is using port $PORT — run: $0 stop"
    exit 1
  fi
  nohup bundle exec jekyll serve --host $HOST --port "$PORT" > "$LOG_FILE" 2>&1 &
  disown
  echo "Starting… logging to $LOG_FILE"
  if wait_up; then echo "Up at http://localhost:$PORT/"; exit 0; fi
  echo "Still not responding after 15s — check $LOG_FILE"
  exit 1
}

# Review mode: a clean build, then `jekyll build --watch` keeps _site current while
# tools/review-server.py serves it with the feedback overlay injected. Notes land in
# FEEDBACK.md / FEEDBACK.json (gitignored). Nothing in the site files is modified.
review() {
  stop_quiet
  clean
  bundle exec jekyll build > "$LOG_FILE" 2>&1 || { echo "Build failed — see $LOG_FILE"; exit 1; }
  nohup bundle exec jekyll build --watch >> "$LOG_FILE" 2>&1 &
  disown
  nohup python3 tools/review-server.py > "$REVIEW_LOG" 2>&1 &
  disown
  if ! wait_up; then echo "Review server not responding — check $REVIEW_LOG"; exit 1; fi
  cat <<EOF

  sushantdaga.com — review mode

    Home       http://localhost:$PORT/
    Career     http://localhost:$PORT/career/
    Writing    http://localhost:$PORT/writing/

  Feedback overlay is on: select any text, or click "Pin a spot" and click
  anywhere, then pick a tag and/or type a note. "Note on page" for whole-page
  comments. "Show all" lists everything; alt+H hides the marks.
  Notes land in FEEDBACK.md — tell Claude when you're done and it will read them.

  Proposed changes to tap Good / Bad:   http://localhost:$PORT/__rv/

  Stop with: ./serve.sh stop
EOF
}

stop_quiet() {
  pkill -f "$PATTERN" 2>/dev/null || true
  pkill -f "$WATCH_PATTERN" 2>/dev/null || true
  pkill -f "$REVIEW_PATTERN" 2>/dev/null || true
  sleep 0.3
}

stop() {
  local any=0
  pgrep -f "$PATTERN" > /dev/null && any=1
  pgrep -f "$WATCH_PATTERN" > /dev/null && any=1
  pgrep -f "$REVIEW_PATTERN" > /dev/null && any=1
  stop_quiet
  if [ "$any" = 1 ]; then echo "Stopped."; else echo "Not running."; fi
}

status() {
  local any=0
  pgrep -f "$PATTERN" > /dev/null && { echo "Jekyll dev server running at http://localhost:$PORT/"; any=1; }
  pgrep -f "$REVIEW_PATTERN" > /dev/null && { echo "Review server (feedback overlay) running at http://localhost:$PORT/"; any=1; }
  pgrep -f "$WATCH_PATTERN" > /dev/null && { echo "jekyll build --watch running"; any=1; }
  [ "$any" = 1 ] || echo "Not running."
}

# Wipes Jekyll's build/cache dirs. (The old "unstyled site" bug was a theme collision,
# fixed in _config.yml with theme: null — see CLAUDE.md "Verifying changes" — but a clean
# rebuild is still the right first move when the build looks wrong.)
clean() {
  rm -rf _site .jekyll-cache .jekyll-metadata
  echo "Cleaned _site, .jekyll-cache, .jekyll-metadata."
}

case "${1:-}" in
  start)   start ;;
  review)  review ;;
  stop)    stop ;;
  restart) stop_quiet; clean; start ;;
  clean)   stop_quiet; clean ;;
  status)  status ;;
  *) echo "Usage: $0 {start|review|stop|restart|status|clean}"; exit 1 ;;
esac
