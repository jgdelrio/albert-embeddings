from aiohttp import web
from json.decoder import JSONDecodeError
from albert_emb.config import *
from albert_emb.utils import logger
from albert_emb.nlp_model import get_embeddings
import json


async def healthcheck(request):
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return web.Response(text=json.dumps("Healthy"), headers=headers, status=200)


async def encode(request):
    content_type = request.headers.get("Content-Type")
    try:
        parameters = await request.json()
    except JSONDecodeError as err:
        logger.error(f"Invalid JSON error: {err}")
        return web.Response(text=json.dumps(f"Invalid JSON error: {err}"), status=400)

    if content_type != "application/json":
        logger.error("Content-Type header must be 'application/json':")
        return web.Response(
            text=json.dumps("Content-Type header must have value of application/json"),
            status=400)

    try:
        text = parameters.get("text", None)
        if text is None:
            logger.info(f"Error: 'text' parameter was not provided")
            raise ValueError("text must be provided")
        elif not isinstance(text, str):
            logger.info(f"Error: 'text' must be a string")
            raise TypeError("text is not string type")

        logger.info(f"Processing text: {text[:15]}...")
        processed_emb = get_embeddings(text, logger=logger)

        # Response including API and model info
        response = {
            "embeddings": processed_emb['embeddings'].cpu().numpy().tolist(),
            "word_count": processed_emb['word_count'],
            "char_count": processed_emb['char_count'],
            "model_name": processed_emb['model_name'],
            "model_version": processed_emb['model_version'],
            "API_version": API_VERSION}

        return web.Response(
            body=json.dumps(response),
            headers=dict({"Content-Type": "application/json"}),
            status=200)
    except Exception as err:
        logger.error(f"Internal server error: {err}")
        return web.Response(text=json.dumps(f"Internal server error: {err}"), status=500)


"""Define app and API endpoints"""
app = web.Application()
app.router.add_get("/embeddings/healthcheck", healthcheck)
app.router.add_post("/embeddings", encode)


if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=(int(PORT)))
