from __future__ import annotations

import json
import os

from flask import Flask, jsonify, render_template, request

from . import core


def create_app(provider_name: str | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(core.BASE_DIR / "templates"),
        static_folder=str(core.BASE_DIR / "static"),
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/recommend")
    def recommend():
        if provider_name:
            os.environ["MODEL_PROVIDER"] = provider_name
        result, status = core.run_agent(request.get_json(force=True))
        return jsonify(result), status

    @app.post("/api/places/sync")
    def sync_places():
        try:
            places = core.sync_place_db()
        except RuntimeError as error:
            return jsonify({"errors": [str(error)]}), 503
        payload = json.loads(core.PLACE_DB_PATH.read_text(encoding="utf-8"))
        return jsonify(
            {
                "count": len(places),
                "db_path": str(core.PLACE_DB_PATH),
                "source": payload.get("source"),
                "source_counts": payload.get("source_counts", {}),
                "sync_errors": payload.get("sync_errors", []),
                "places": places,
            }
        )

    return app
