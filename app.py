# === FASTAPII ATENSIÓN ===
import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Numeric, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from typing import Union

# 🎯 Conexión a BD
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === Modelo de tabla ===
class HTARegistro(Base):
    __tablename__ = "hta_registros"
    id = Column(Integer, primary_key=True, index=True)
    sexo = Column(String(10), nullable=False)
    edad = Column(Integer, nullable=False)
    peso = Column(Numeric(5, 2), nullable=False)
    altura = Column(Numeric(5, 2), nullable=False)
    bmi = Column(Numeric(5, 2), nullable=False)
    sal = Column(String(5), nullable=False)
    alcohol = Column(String(5), nullable=False)
    tabaco = Column(String(30), nullable=False)
    vapeo = Column(String(30), nullable=False)
    estres_dias = Column(Integer, nullable=False)
    actividad = Column(String(5), nullable=False)
    colesterol = Column(String(5), nullable=False)
    diabetes = Column(String(30), nullable=False)
    diagnosticado_hta = Column(String(5), nullable=False)
    riesgo = Column(String(10), nullable=False)
    probabilidad = Column(Numeric(5, 2), nullable=False)
    puntaje_conocimiento_hta = Column(Integer)
    respuestas_hta = Column(String)
    fecha_registro = Column(DateTime(timezone=True), server_default=func.now())

# ✅ Crear tabla si no existe
def crear_tablas_si_no_existen():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print("❌ Error creando tablas:", e)

crear_tablas_si_no_existen()

# === App y modelo ===
app = FastAPI()
modelo = joblib.load("modelo_rf_actualizado.pkl")

# === Esquemas de entrada ===
class EntradaPrediccion(BaseModel):
    sexo: int
    edad: int
    peso: float
    altura: float
    sal: int
    alcohol: int
    tabaco: int
    vapeo: int
    estres_dias: int
    actividad: int
    colesterol: int
    diabetes: int

class EntradaCompleta(EntradaPrediccion):
    diagnosticado_hta: Union[int, bool]
    puntaje: int
    respuestas: dict

# === Funciones auxiliares ===
def interpretar(prob):
    if prob >= 0.65:
        return "Alto"
    elif prob >= 0.35:
        return "Moderado"
    else:
        return "Bajo"

def codificar_edad(edad):
    if 18 <= edad <= 24: return 1
    elif edad <= 29: return 2
    elif edad <= 34: return 3
    elif edad <= 39: return 4
    elif edad <= 44: return 5
    elif edad <= 49: return 6
    elif edad <= 54: return 7
    elif edad <= 59: return 8
    elif edad <= 64: return 9
    elif edad <= 69: return 10
    elif edad <= 74: return 11
    elif edad <= 79: return 12
    elif edad <= 100: return 13
    else: return 0

def texto_sexo(s): return "Hombre" if s == 1 else "Mujer"
def texto_binario(v): return "Sí" if v == 1 else "No"
def texto_tabaco(v):
    return {
        1: "Fumador diario", 2: "Fumador ocasional", 3: "Exfumador", 4: "No fumador"
    }.get(v, "Desconocido")

def texto_vapeo(v):
    return {
        1: "Todos los días", 2: "Algunos días", 3: "Raramente", 4: "Nunca he usado"
    }.get(v, "Desconocido")

def texto_diabetes(v):
    return {
        0: "No tengo diagnóstico", 1: "Tengo diagnóstico", 2: "No sé"
    }.get(v, "Desconocido")

# === Endpoint: Solo predicción ===
@app.post("/predict")
def predecir_riesgo(data: EntradaPrediccion):
    try:
        bmi = round(data.peso / ((data.altura / 100) ** 2), 2)
        entrada = [
            codificar_edad(data.edad),
            data.sexo,
            bmi,
            data.estres_dias,
            data.sal,
            data.actividad,
            data.tabaco,
            data.vapeo,
            data.alcohol,
            data.diabetes,
            data.colesterol
        ]
        prob = modelo.predict_proba([entrada])[0][1]
        nivel = interpretar(prob)

        return {
            "riesgo": nivel,
            "probabilidad": round(prob * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

# === Endpoint: Guardar evaluación completa ===
@app.post("/guardar")
def guardar_valoracion(data: EntradaCompleta):
    try:
        bmi = round(float(data.peso) / ((float(data.altura) / 100) ** 2), 2)
        entrada = [
            int(codificar_edad(data.edad)),
            int(data.sexo),
            float(bmi),
            int(data.estres_dias),
            int(data.sal),
            int(data.actividad),
            int(data.tabaco),
            int(data.vapeo),
            int(data.alcohol),
            int(data.diabetes),
            int(data.colesterol)
        ]
        prob = float(modelo.predict_proba([entrada])[0][1])
        nivel = interpretar(prob)

        db = SessionLocal()
        nuevo = HTARegistro(
            diagnosticado_hta=str(texto_binario(data.diagnosticado_hta)),
            sexo=str(texto_sexo(data.sexo)),
            edad=int(data.edad),
            peso=float(data.peso),
            altura=float(data.altura),
            bmi=float(bmi),
            sal=str(texto_binario(data.sal)),
            alcohol=str(texto_binario(data.alcohol)),
            actividad=str(texto_binario(data.actividad)),
            tabaco=str(texto_tabaco(data.tabaco)),
            vapeo=str(texto_vapeo(data.vapeo)),
            colesterol=str(texto_binario(data.colesterol)),
            diabetes=str(texto_diabetes(data.diabetes)),
            estres_dias=int(data.estres_dias),
            riesgo=str(nivel),
            probabilidad=round(float(prob) * 100, 2),
            puntaje_conocimiento_hta=int(data.puntaje),
            respuestas_hta=str(data.respuestas)
        )
        db.add(nuevo)
        db.commit()
        db.close()

        return {
            "mensaje": "✅ Datos guardados correctamente",
            "riesgo": nivel,
            "probabilidad": round(prob * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
