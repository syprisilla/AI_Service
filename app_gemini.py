from __future__ import annotations

import os
import logging

os.environ["MODEL_PROVIDER"] = "gemini"
logging.basicConfig(level=logging.INFO)

import uvicorn
from uvicorn.middleware.wsgi import WSGIMiddleware

from trip_agent.web import create_app


flask_app = create_app("gemini")
app = WSGIMiddleware(flask_app)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)
