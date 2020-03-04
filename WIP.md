https://hub.docker.com/r/bitnami/pytorch/          2.45Gb torch 1.4


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