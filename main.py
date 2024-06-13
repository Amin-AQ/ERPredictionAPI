import logging
from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import models, schemas
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
from pipeline import generate_prediction
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Set up CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add a user and return boolean value
@app.post("/add_user/", response_model=dict)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        return {"success": False}
    new_user = models.User(username=user.username, email=user.email, password=user.password, isAdmin=user.isAdmin)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"success": True}

# Get a user and return boolean values foundUser and isAdmin
@app.post("/login/", response_model=dict)
def login_user(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email, models.User.password == user.password).first()
    if db_user:
        return {"foundUser": True, "username":db_user.username, "isAdmin": db_user.isAdmin}
    return {"foundUser": False}

# Add a patient and return boolean value
@app.post("/add_patient/", response_model=dict)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    new_patient = models.Patient(**patient.dict())
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return {"success": True}

@app.get("/patients/", response_model=List[schemas.Patient])
def get_patients(skip: int = 0, db: Session = Depends(get_db)):
    patients = db.query(models.Patient).offset(skip).all()
    return patients


@app.get("/users/", response_model=List[schemas.User])
def get_users(skip: int = 0, db: Session = Depends(get_db)):
    users = db.query(models.User).offset(skip).all()
    return users

@app.get("/reports/", response_model=List[schemas.Report])
def get_reports(skip: int = 0, db: Session = Depends(get_db)):
    reports = db.query(models.Report).offset(skip).all()
    return reports

@app.post("/add_report/", response_model=schemas.Report)
async def add_report(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print(f"Received patient_id: {patient_id}")
    print(f"Received file: {file.filename}")

    os.makedirs('images', exist_ok=True)

    # Save the uploaded file
    file_path = f'images/{file.filename}'
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    final_prediction = generate_prediction(file.filename)
    # generate report here
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    # Create a new report entry in the database
    report = models.Report(
        patient_id=patient_id,
        date=datetime.now(),
        fileName=file.filename  # this is image file name, not report pdf name it will be inferred from report id and patient 
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    make_report(patient,report,final_prediction)

    return report

@app.delete("/delete_user/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.put("/reset_password/{user_id}")
async def update_password(user_id: int, password_data: schemas.UserPasswordUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password = password_data.new_password
    db.commit()
    return {"message": "Password updated successfully"}

@app.put("/reset_password_by_email")
async def update_password_by_email(password_data: schemas.UserPasswordUpdateByEmail, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == password_data.email).first()
    if not user:
        return {"success":False}
    user.password = password_data.new_password
    db.commit()
    return {"success": True}

@app.get("/download_report/{report_id}/{patient_id}")
async def download_report(report_id: int, patient_id: int):
    file_path = f"./pdf_reports/{report_id}_{patient_id}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(file_path, media_type='application/pdf')

# Function to add an admin user manually
def add_admin_user():
    db = SessionLocal()
    admin_name="amin"
    admin_email = "admin1@example.com"
    admin_password = "adminpassword"
    admin_user = db.query(models.User).filter(models.User.email == admin_email).first()
    if not admin_user:
        new_admin = models.User(username=admin_name, email=admin_email, password=admin_password, isAdmin=True)
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
    db.close()

# Add admin user when the server starts
add_admin_user()


def make_report(patient, report, pred):
    final_prediction = "Positive" if pred == 1 else "Negative"
    
    # Generate PDF report name as reportid_patientid.pdf
    pdf_report_name = f"{report.id}_{patient.id}.pdf"
    pdf_path = f'pdf_reports/{pdf_report_name}'

    c = canvas.Canvas(pdf_path, pagesize=letter)
    page_width, page_height = letter

    # Center the image while maintaining aspect ratio
    image_path = "./assets/logo.png"
    image_width, image_height = 130, 100
    image_x = (page_width - image_width) / 2 
    image_y = 700  
    c.drawImage(image_path, image_x, image_y, width=image_width, height=image_height, preserveAspectRatio=True)

    # Patient details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 670, "Patient Details")
    c.setFont("Helvetica", 12)
    c.drawString(50, 650, f"Name: {patient.name}")
    c.drawString(50, 630, f"Age: {patient.age}")
    c.drawString(50, 610, f"Blood Group: {patient.blood_group}")
    c.drawString(50, 590, f"Contact: {patient.contact}")
    c.drawString(50, 570, f"Address: {patient.address}")

    # Report details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 550, "Report Details")
    c.setFont("Helvetica", 12)

     # Format date with day and month suffix
    day = report.date.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]

    formatted_date = report.date.strftime(f"%A, {day}{suffix} %B, %Y %I:%M%p").lower().replace("am", "am").replace("pm", "pm")
    parts = formatted_date.split(' ')
    parts[0] = parts[0].capitalize() 
    parts[2] = parts[2].capitalize()  
    formatted_date = ' '.join(parts)

    c.drawString(50, 530, f"Report ID: {report.id}")
    c.drawString(50, 510, f"Date: {formatted_date}")
    
    # Final Prediction in Bold
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 490, f"ER Status Prediction: {final_prediction}")

    # Footer
    footer_text = "ER Prediction System | Lahore, Pakistan | aminqasmi78@gmail.com"
    c.setFont("Helvetica", 10)
    text_width = c.stringWidth(footer_text, "Helvetica", 10)
    c.drawString((page_width - text_width) / 2, 50, footer_text)  # Centering the footer text horizontally

    c.save()
    return pdf_report_name