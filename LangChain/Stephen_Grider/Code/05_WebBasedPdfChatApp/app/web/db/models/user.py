import uuid
from app.web.db import db
from .base import BaseModel


class User(BaseModel):
    id: str = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: str = db.Column(db.String(80), unique=True, nullable=False)
    password: str = db.Column(db.String(80), nullable=False)
    pdfs = db.relationship("Pdf", back_populates="user")
    conversations = db.relationship("Conversation", back_populates="user")

    def as_dict(self):
        return {"id": self.id, "email": self.email}
