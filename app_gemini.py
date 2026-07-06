from __future__ import annotations

import os

os.environ["MODEL_PROVIDER"] = "gemini"

from trip_agent.web import create_app


app = create_app("gemini")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5001)
