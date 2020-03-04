# ALBERT Embeddings as Service

Provides an API to generate embeddings from the text provided.

The default model is ALBERT-base-v2 which is a bidirectional embeddings model.

More info at: [ALBERT](https://github.com/google-research/ALBERT) and [github](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html).


## API

There are two end-points:
- /embeddings

- /embeddings/healthcheck

'embeddings' accepts a POST request with text from which the embeddings are to be generated.

'healthcheck' accepts a GET request.


## Curl request examples:

Note: Update the port to 8080 if using the Dockerfile

The most simple request using the default options:

curl -X POST http://localhost:8080/embeddings



