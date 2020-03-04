#!/bin/sh

# Prepare log files and start outputting logs to stdout
touch /albert_repo/gunicorn.log
touch /albert_repo/access.log
echo Starting nginx
# Start Gunicorn processes
echo Starting Gunicorn
exec gunicorn main:app \
    --bind localhost:5000 \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --workers 1 \
    --log-level=info \
    --log-file=/albert_repo/gunicorn.log \
    --access-logfile=/albert_repo/access.log &

exec nginx -g "daemon off;"



# Log to stdout/stderr by default
#if [ -n "$LOG_FOLDER" ]; then
#    ACCESS_LOG=${LOG_FOLDER}/access.log
#    ERROR_LOG=${LOG_FOLDER}/error.log
#else
#    ACCESS_LOG=/proc/1/fd/1
#    ERROR_LOG=/proc/1/fd/2
#fi
#
#echo Starting nginx
#
# Start Gunicorn processes
#echo Starting Gunicorn
#exec gunicorn main:app \
#    --bind localhost:5000 \
#    --worker-class aiohttp.worker.GunicornWebWorker \
#    --workers 1 \
#    --log-level=info \
#    --access-logfile "$ACCESS_LOG" \
#    --error-logfile "$ERROR_LOG"

#exec nginx -g "daemon off;"