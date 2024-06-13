from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str 
    email: str
    isAdmin: bool = False

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserPasswordUpdate(BaseModel):
    new_password: str

class UserPasswordUpdateByEmail(BaseModel):
    email: str
    new_password: str

class User(UserBase):
    id: int

    class Config:
        orm_mode = True

class PatientBase(BaseModel):
    name: str
    age: int
    blood_group: str
    contact: str
    address: str

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int

    class Config:
        orm_mode = True

class ReportBase(BaseModel):
    date: datetime
    fileName: str

class ReportCreate(ReportBase):
    patient_id: int

class Report(ReportBase):
    id: int
    patient: Patient

    class Config:
        orm_mode = True

