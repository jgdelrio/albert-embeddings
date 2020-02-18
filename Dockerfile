FROM python:3.7-alpine

RUN apk add --no-cache nginx

WORKDIR /albert_emb

COPY /albert_emb/requirements.txt /albert_emb

RUN pip install --no-deps -r requirements.txt

COPY /albert_emb /albert_emb

COPY docker_launch.sh /

COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p /run/nginx && nginx -t

EXPOSE 8080

ENTRYPOINT ["/docker_launch.sh"]
