from invoke import task
from sympy import false


@task
def dev(ctx):
    ctx.run(
        "flask --app app.web run --debug --port 8000",
        pty=false,
        env={"APP_ENV": "development"},
    )


@task
def devworker(ctx):
    ctx.run(
        "watchmedo auto-restart --directory=./app --pattern=*.py --recursive -- celery -A app.celery.worker worker --concurrency=1 --loglevel=INFO",
        pty=false,
        env={"APP_ENV": "development"},
    )
