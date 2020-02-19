FROM python:3.7.6

RUN apt-get update \
    && apt-get install -y \
        nginx \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt /albert_emb/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-deps -r /albert_emb/requirements.txt

WORKDIR /albert_emb

COPY . /albert_emb

COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN chmod 777 -R /albert_emb/docker_launch.sh

RUN mkdir -p /run/nginx && nginx -t

#run python3 -c 'from albert_emb.nlp_model import retrieve_model; retrieve_model()'

EXPOSE 8080

ENTRYPOINT ["/albert_emb/docker_launch.sh"]
