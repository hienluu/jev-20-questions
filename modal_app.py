"""Modal deployment: `modal deploy modal_app.py`.

Expects a Modal secret named `typesafe` holding TYPESAFE_API_KEY:
    modal secret create typesafe --from-dotenv .env
Build the matrix first so it ships with the image.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("app", "twenty_q")
    .add_local_dir("twenty_q/data", "/root/twenty_q/data")
    .add_local_dir("static", "/root/static")
)

app = modal.App("twenty-questions-jev", image=image)  # `modal serve/deploy` looks for `app`


@app.function(secrets=[modal.Secret.from_name("typesafe")])
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    from app import app as fastapi_app  # the FastAPI app in app.py

    return fastapi_app
