#!/bin/sh
# Stage the app into the personal site at /instrumental.
#
# The page is static; the compute lives on the Mac mini behind a Cloudflare
# tunnel, whose hostname is written into config.js here so the checked-in copy
# stays pointed at localhost for development.
set -e
cd "$(dirname "$0")"

API="${INSTRUMENTAL_API:-https://api.philippbogdan.com}"
DEST=~/personal-website/public/instrumental

mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.DS_Store' \
  --exclude 'server.py' \
  --exclude 'stats.html' \
  --exclude '__pycache__' \
  app/ "$DEST/"

cat > "$DEST/config.js" <<EOF
/* Written by publish.sh. The matching runs on the Mac mini behind this host. */
window.INSTRUMENTAL_API = '$API';
EOF

echo "staged $DEST  ($(du -sh "$DEST" | cut -f1), api=$API)"
