#!/bin/bash

# Disabled: this fork does not publish or deploy yet. Releases are made on
# GitHub. Remove these lines to re-enable.
echo "sync-web.sh is disabled for this fork." >&2
exit 1
for f in examples/*.{html,js,css}; do
  gzip -k $f
done
rsync -vLr examples smartcrop.js doc/example.jpg x.29a.ch:/var/www/static/sandbox/2014/smartcrop/
rm examples/*.gz
