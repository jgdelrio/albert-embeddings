#!/bin/sh

# Prepare log files and start outputting logs to stdout
touch /logs/gunicorn.log
touch /logs/access.log
echo Starting nginx
# Start Gunicorn processes
echo Starting Gunicorn
exec gunicorn service:app \
    --bind localhost:5000 \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --workers 1 \
    --log-level=info \
    --log-file=/logs/gunicorn.log \
    --access-logfile=/logs/access.log &

exec nginx -g "daemon off;"