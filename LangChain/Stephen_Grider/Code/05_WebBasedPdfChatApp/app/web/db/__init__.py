import click
import os
from flask_sqlalchemy import SQLAlchemy
from flask import current_app

db = SQLAlchemy()


@click.command("init-db")
def init_db_command():
    with current_app.app_context():
        try:
            os.makedirs(current_app.instance_path)
        except OSError:
            pass
        db.drop_all()
        db.create_all()
    click.echo("Initialized the database.")
