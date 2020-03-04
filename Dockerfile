FROM anibali/pytorch:no-cuda

WORKDIR /

COPY requirements.txt /albert_repo/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-deps -r /albert_repo/requirements.txt

WORKDIR /albert_repo

COPY . /albert_repo

USER root
RUN mkdir -p /run/nginx
RUN apt-get update \
    && apt-get install -y \
        nginx \
    && rm -rf /var/lib/apt/lists/*

COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN nginx -t

#run python3 -c 'from albert_emb.nlp_model import retrieve_model; retrieve_model()'

EXPOSE 8080

ENTRYPOINT ["/albert_repo/docker_launch.sh"]
