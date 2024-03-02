from typing import TypeVar, Optional, Type, List
from abc import abstractmethod
from app.web.db import db

T = TypeVar("T", bound="BaseModel")


class BaseModel(db.Model):
    __abstract__ = True

    @classmethod
    def create(cls: Type[T], commit=True, **kwargs) -> T:
        instance = cls(**kwargs)
        return instance.save(commit)

    @classmethod
    def find_by(cls: Type[T], **kwargs) -> Optional[T]:
        return db.session.execute(db.select(cls).filter_by(**kwargs)).scalar_one()

    @classmethod
    def where(cls: Type[T], **kwargs) -> List[T]:
        return db.session.execute(db.select(cls).filter_by(**kwargs)).scalars().all()

    @classmethod
    def upsert(cls: Type[T], commit=True, **kwargs) -> T:
        instance = None
        if kwargs.get("id"):
            instance = cls.find_by(id=kwargs["id"])

        if instance:
            instance.update(commit, **kwargs)
            return instance
        else:
            instance = cls.create(**kwargs)
            return instance

    @classmethod
    def delete_by(cls, commit: bool = True, **kwargs) -> None:
        instance = cls.find_by(**kwargs)
        db.session.delete(instance)
        if commit:
            return db.session.commit()
        return

    @classmethod
    def as_dicts(cls, models):
        return [m.as_dict() for m in models]

    @abstractmethod
    def as_dict(self):
        raise NotImplementedError

    def update(self, commit=True, **kwargs):
        for attr, value in kwargs.items():
            if attr != ["id"]:
                setattr(self, attr, value)
        if commit:
            return self.save()
        return self

    def save(self, commit=True):
        db.session.add(self)
        if commit:
            db.session.commit()
        return self
