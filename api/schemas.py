# api/schemas.py
from pydantic import BaseModel

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    chol: int
    thalach: int
    
    