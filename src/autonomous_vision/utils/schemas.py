from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_gt_x1(cls, v, info):
        x1 = info.data.get("x1")
        if x1 is not None and v <= x1:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def y2_gt_y1(cls, v, info):
        y1 = info.data.get("y1")
        if y1 is not None and v <= y1:
            raise ValueError("y2 must be > y1")
        return v


class ObjectAttributes(BaseModel):
    id: int
    category: str
    box2d: Optional[BBox] = None
    attributes: Optional[dict] = None


class ImageAttributes(BaseModel):
    scene: Optional[str] = None
    timeofday: Optional[str] = None
    weather: Optional[str] = None


class ImageAnnotation(BaseModel):
    name: str
    attributes: Optional[ImageAttributes] = None
    labels: List[ObjectAttributes] = Field(default_factory=list)
