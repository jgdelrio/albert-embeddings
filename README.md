# ALBERT Embeddings as Service

Provides an API to generate embeddings from the text provided.

The default model is ALBERT-base-v2 which is a bidirectional embeddings model.

More info about [ALBERT](https://github.com/google-research/ALBERT) and at [github](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html).


## API

There are two end-points:
- /embeddings

- /embeddings/healthcheck

'embeddings' accepts a POST request with text from which the embeddings are to be generated.

'healthcheck' accepts a GET request.


Curl request examples:
Note: Update the port to 8080 if using the Dockerfile

The most simple request using the default options:

curl -X POST http://localhost:8080/embeddings



## Next developments
- Docker image reduction:

https://hub.docker.com/r/bitnami/pytorch/          2.45Gb torch 1.4

docker pull anibali/pytorch:no-cuda                

Code to evaluate:

Or Python alpine with cleaning:
RUN echo "|--> Updating" \
    && apk update && apk upgrade \
    && echo "|--> Install PyTorch" \
    && git clone --recursive https://github.com/pytorch/pytorch \
    && cd pytorch && python setup.py install \
    && echo "|--> Install Torch Vision" \
    && git clone --recursive https://github.com/pytorch/vision \
    && cd vision && python setup.py install \
    && echo "|--> Cleaning" \
    && rm -rf /pytorch \
    && rm -rf /root/.cache \
    && rm -rf /var/cache/apk/* \
    && apk del .build-deps \
    && find /usr/lib/python3.6 -name __pycache__ | xargs rm -r \
    && rm -rf /root/.[acpw]*
    
    
Or multi-build stage:
https://medium.com/the-artificial-impostor/smaller-docker-image-using-multi-stage-build-cb462e349968

# Instal basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget unzip bzip2 build-essential ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*