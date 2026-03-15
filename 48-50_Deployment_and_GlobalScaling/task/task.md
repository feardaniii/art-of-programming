# Deployment & Global Scaling - Tasks
## Sesiunile 48-50 / Sessions 48-50

---

## Context: Ce avem deja / What we already have

Din sesiunile anterioare ai deja:
- `health_predictor_production.keras` + `health_scaler.pkl` (model antrenat)
- `app.py` (FastAPI cu `/predict`, `/predict/batch`, `/metrics`, `/health`)
- `1_streamlit_diabetes.py` (Streamlit UI)
- `cat_vs_dog_classifier.keras` (CNN din sesiunea 37-38)

Aceste task-uri construiesc pe munca anterioara!

---

## Postman vs FastAPI /docs: De ce ambele?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   FastAPI /docs (Swagger UI)          Postman                          │
│   ──────────────────────────          ───────                          │
│   - Vine GRATIS cu FastAPI            - Tool separat (descarca-l)      │
│   - Perfect pt. testare rapida        - Salveaza request-uri           │
│   - Se actualizeaza automat           - Organizeaza in "Collections"   │
│   - Nu poti salva teste               - Poate testa ORICE API          │
│   - Dispare cand opresti serverul     - Exporta/importa teste          │
│   - Doar pt. FastAPI                  - Variabile de environment       │
│                                       - Automatizare cu teste          │
│                                                                         │
│   CONCLUZIE:                                                           │
│   FastAPI /docs = "taste rapid in timp ce dezvolt"                     │
│   Postman = "test suite profesional pe care il pastrez si il impart"   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Gandeste-te asa:
  FastAPI /docs = Notepad    (rapid, temporar)
  Postman       = Word       (salvezi, organizezi, trimiti)
```

**Download Postman:** https://www.postman.com/downloads/

---

## Docker: Filosofie in 30 de secunde

```
PROBLEMA:  "La mine merge!" (dar la colegul tau nu)

SOLUTIA:   Docker impacheteaza TOTUL:
           codul tau + Python + TensorFlow + dependinte
           intr-o "cutie" (container) care merge ORIUNDE.

ANALOGIE:  Docker = un container de transport.
           Nu conteaza ce e inauntru (cafea, masini, laptopuri).
           Containerul se potriveste pe orice camion, tren sau vapor.
           La fel, un Docker container ruleaza pe orice masina.

┌──────────────────────────────────┐
│  Dockerfile = reteta            │
│  Image = cutia impachetata       │
│  Container = cutia care ruleaza  │
└──────────────────────────────────┘
```

**Dockerfile Starter (pentru FastAPI + Keras):**

```dockerfile
# 1. Porneste de la o imagine Python oficiala
FROM python:3.11-slim

# 2. Seteaza directorul de lucru IN container
WORKDIR /app

# 3. Copiaza fisierele de dependinte
COPY requirements.txt .

# 4. Instaleaza dependintele
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiaza TOATA aplicatia (cod + model + scaler)
COPY . .

# 6. Expune portul pe care asculta FastAPI
EXPOSE 8000

# 7. Comanda care porneste aplicatia
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**
```
fastapi
uvicorn
tensorflow
joblib
numpy
pydantic
```

**Comenzi esentiale:**
```bash
docker build -t my-ml-api .          # Construieste imaginea
docker run -p 8000:8000 my-ml-api    # Ruleaza containerul
# Acum mergi la http://localhost:8000/docs
```

---

## Kubernetes: Ce e si de ce exista (explicat simplu)

```
PROBLEMA:
  Ai 1 container Docker cu API-ul tau. Vine Black Friday.
  1 container nu face fata la 10.000 de cereri pe secunda.

SOLUTIA:  Kubernetes (prescurtat "K8s")
  - Porneste automat MAI MULTE copii ale containerului tau
  - Daca unul crapa, il reporneste automat
  - Distribuie cererile intre copii (load balancing)
  - Scaleaza in sus/jos dupa trafic

ANALOGIE:
  Docker = un chelner care serveste mese
  Kubernetes = MANAGERUL restaurantului care:
    - Aduce mai multi chelneri cand e plin
    - Trimite chelneri acasa cand e gol
    - Inlocuieste un chelner care s-a imbolnavit
    - Se asigura ca fiecare masa e servita rapid

  Tu nu ii zici lui Kubernetes "porneste 5 containere".
  Tu ii zici "vreau sa pot servi 1000 cereri/secunda"
  si el decide CATE containere are nevoie.

IN PRACTICA (pentru noi acum):
  - Docker = suficient pentru proiecte mici-medii
  - Kubernetes = cand ai trafic mare si nevoie de reliability
  - NU trebuie sa stii Kubernetes acum, dar e bine sa stii ce face
```

---

### Task 1: Flask Image Prediction API (Session 48 - Warmup)
**Dificultate / Difficulty: Easy | Timp / Time: 45 min**

**RO:** Construieste un API cu Flask care primeste o imagine si
returneaza predictia. Foloseste modelul Fashion-MNIST (cel din
sesiunile 37-38) sau orice alt model de clasificare imagini pe care
l-ai antrenat. API-ul trebuie sa aiba un endpoint POST `/predict`
care primeste o imagine si returneaza clasa si confidenta.

**EN:** Build a Flask API that receives an image and returns the
prediction. Use your Fashion-MNIST model (from sessions 37-38) or
any other image classification model you trained. The API needs a
POST `/predict` endpoint that accepts an image and returns the class
and confidence.

**Cod de start / Starter Code:**

```python
from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from PIL import Image
import io

app = Flask(__name__)

# Incarca modelul (antrenat in sesiunile 37-38)
model = keras.models.load_model('your_model.keras')

CLASS_NAMES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@app.route('/predict', methods=['POST'])
def predict():
    # TODO: Primeste imaginea din request
    # TODO: Preproceseaza (resize, normalize, reshape)
    # TODO: Fa predictia cu model.predict()
    # TODO: Returneaza JSON cu clasa si confidenta
    pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "Fashion-MNIST"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Hint: Cum primesti o imagine in Flask:**
```python
file = request.files['image']
img = Image.open(io.BytesIO(file.read()))
img = img.convert('L').resize((28, 28))       # Grayscale, 28x28
img_array = np.array(img) / 255.0             # Normalize
img_array = img_array.reshape(1, 28, 28, 1)   # Batch shape
```

**Hint: Cum testezi cu Python (fara Postman inca):**
```python
import requests
response = requests.post(
    'http://localhost:5000/predict',
    files={'image': open('test_shoe.png', 'rb')}
)
print(response.json())
# {"class": "Sneaker", "confidence": 0.94, "all_probabilities": {...}}
```

**Cerinte / Requirements:**
- Endpoint POST `/predict` care primeste o imagine
- Returneaza JSON cu: `class`, `confidence`, `all_probabilities`
- Endpoint GET `/health` care returneaza statusul
- Trateaza erori (imagine lipsa, format invalid)
- Testeaza cu minimum 3 imagini diferite
- Compara cu FastAPI: care e mai usor? Care are docs automate?

**Evaluare / Evaluation:**
- [ ] Flask app porneste fara erori
- [ ] `/health` returneaza JSON corect
- [ ] `/predict` accepta o imagine si returneaza predictia
- [ ] JSON-ul contine class, confidence, all_probabilities
- [ ] Erori tratate (imagine lipsa returneaza mesaj clar)
- [ ] Testat cu 3+ imagini
- [ ] Scris 3+ propozitii: diferente Flask vs FastAPI

---

### Task 2: Dockerfile pentru Aplicatia ta (Session 49 - Core)
**Dificultate / Difficulty: Easy-Medium | Timp / Time: 45 min**

**RO:** Containerizeaza aplicatia Flask din Task 1 (sau FastAPI-ul
din sesiunea 47) folosind Docker. Scrie un Dockerfile, construieste
imaginea, ruleaz-o si verifica ca API-ul functioneaza din container.

**EN:** Containerize your Flask app from Task 1 (or the FastAPI from
session 47) using Docker. Write a Dockerfile, build the image, run
it and verify the API works from inside the container.

**Cerinte / Requirements:**
- Scrie un `Dockerfile` (foloseste starter-ul de mai sus ca baza)
- Scrie `requirements.txt` cu toate dependintele
- Construieste imaginea: `docker build -t my-ml-api .`
- Ruleaza containerul: `docker run -p 8000:8000 my-ml-api`
- Verifica ca `/health` si `/predict` functioneaza
- Documenteaza dimensiunea imaginii (`docker images`)
- BONUS: Adauga un `.dockerignore` (ce NU vrei in container?)

```
# .dockerignore - fisiere care NU intra in container
__pycache__/
*.pyc
.git/
.env
*.png
*.jpg
README.md
```

**Verificare pas cu pas / Step by step verification:**
```bash
# 1. Construieste
docker build -t my-ml-api .

# 2. Verifica imaginea
docker images | grep my-ml-api

# 3. Ruleaza
docker run -p 8000:8000 my-ml-api

# 4. Intr-un alt terminal, testeaza
curl http://localhost:8000/health

# 5. Opreste
docker ps                    # gaseste CONTAINER_ID
docker stop <CONTAINER_ID>
```

**Evaluare / Evaluation:**
- [ ] Dockerfile scris corect (toate cele 7 instructiuni)
- [ ] requirements.txt complet
- [ ] `docker build` reuseste fara erori
- [ ] `docker run` porneste serverul
- [ ] API-ul raspunde din container (health check OK)
- [ ] `/predict` functioneaza din container
- [ ] Documentat dimensiunea imaginii Docker
- [ ] .dockerignore creat (BONUS)

---

### Task 3: Postman - Test Suite Profesional (Session 49 - Core)
**Dificultate / Difficulty: Medium | Timp / Time: 60 min**

**RO:** Instaleaza Postman si creeaza o colectie completa de teste
pentru API-ul tau (Flask sau FastAPI). Testeaza toate endpoint-urile,
salveaza request-urile intr-o colectie organizata si compara
experienta cu FastAPI /docs (Swagger UI).

**EN:** Install Postman and create a complete test collection for your
API (Flask or FastAPI). Test all endpoints, save requests in an organized
collection and compare the experience with FastAPI /docs (Swagger UI).

**Pasi / Steps:**
1. Descarca Postman de la https://www.postman.com/downloads/
2. Creeaza o colectie noua: "ML API Tests"
3. Adauga request-uri pentru FIECARE endpoint

**Request-uri de creat / Requests to create:**

```
Collection: "ML API Tests"
├── Health Check          (GET  /health)
├── Single Prediction     (POST /predict)
├── Batch Prediction      (POST /predict/batch)      ← daca ai FastAPI
├── Model Metrics         (GET  /metrics)             ← daca ai FastAPI
├── Error: Missing Data   (POST /predict cu body gol)
└── Error: Invalid Data   (POST /predict cu date invalide)
```

**Exemplu body JSON pentru Postman (Health Prediction API):**
```json
{
    "age": 0.05,
    "sex": 0.05,
    "bmi": 0.06,
    "bp": 0.02,
    "s1": -0.04,
    "s2": -0.03,
    "s3": 0.00,
    "s4": -0.03,
    "s5": 0.01,
    "s6": -0.02
}
```

**Cerinte / Requirements:**
- Minimum 5 request-uri salvate in colectie
- Include cel putin 2 cazuri de eroare (body gol, date invalide)
- Screenshot cu colectia Postman organizata
- Screenshot cu un request reusit (status 200 + response body)
- Screenshot cu un request esuat (status 400/500 + error message)
- Scrie comparatie: Postman vs FastAPI /docs (minim 5 propozitii)

**Tabel comparativ de completat:**
```
| Criteriu              | FastAPI /docs    | Postman          |
|-----------------------|------------------|------------------|
| Setup necesar         | ??               | ??               |
| Salvare teste         | ??               | ??               |
| Testare alte API-uri  | ??               | ??               |
| Variabile environment | ??               | ??               |
| Colaborare in echipa  | ??               | ??               |
| Favoritul meu         | ??               | ??               |
```

**Evaluare / Evaluation:**
- [ ] Postman instalat si functional
- [ ] Colectie creata cu 5+ request-uri
- [ ] Health check testat cu succes
- [ ] Predictie testat cu succes
- [ ] 2+ cazuri de eroare testate
- [ ] Screenshot-uri incluse (colectie, succes, eroare)
- [ ] Tabel comparativ completat
- [ ] Comparatie scrisa Postman vs /docs (5+ propozitii)

---

### Task 4: Docker Compose - App + Logging (Session 50 - Advanced)
**Dificultate / Difficulty: Medium-Hard | Timp / Time: 75 min**

**RO:** Foloseste `docker-compose` pentru a rula aplicatia ta
impreuna cu un serviciu de logging. Creeaza doua containere:
(1) API-ul tau de ML si (2) un container care salveaza log-urile
predictiilor intr-un fisier persistent.

**EN:** Use `docker-compose` to run your application together with
a logging service. Create two containers: (1) your ML API and
(2) a container that saves prediction logs to a persistent file.

**Cod de start / Starter Code:**

`docker-compose.yml`:
```yaml
version: '3.8'

services:
  # Serviciul 1: API-ul de ML
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs      # Logs persistente (raman dupa restart)
    environment:
      - MODEL_NAME=health_predictor
      - LOG_LEVEL=INFO

  # Serviciul 2: Un container simplu care monitorizeaza log-urile
  log-viewer:
    image: busybox
    volumes:
      - ./logs:/logs:ro        # Read-only access la logs
    command: sh -c "tail -f /logs/predictions.log"
    depends_on:
      - ml-api
```

**Ce trebuie modificat in app.py (logging):**
```python
import logging
import os
from datetime import datetime

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# In endpoint-ul /predict, dupa predictie:
logging.info(f"Prediction: risk={risk_probability:.3f} | level={risk_level}")
```

**Cerinte / Requirements:**
- `docker-compose.yml` cu minimum 2 servicii
- API-ul sa logheze fiecare predictie intr-un fisier
- Log-urile sa fie persistente (sa ramana dupa `docker-compose down`)
- Testeaza cu `docker-compose up --build`
- Fa 5+ predictii si verifica log-urile
- Opreste cu `docker-compose down` si verifica ca log-urile au ramas

**Comenzi utile:**
```bash
docker-compose up --build        # Construieste si porneste tot
docker-compose up -d             # Porneste in background
docker-compose logs ml-api       # Vezi log-urile API-ului
docker-compose logs log-viewer   # Vezi log viewer-ul
docker-compose down              # Opreste tot
docker-compose ps                # Statusul serviciilor
```

**Evaluare / Evaluation:**
- [ ] docker-compose.yml scris corect
- [ ] `docker-compose up` porneste ambele servicii
- [ ] API-ul raspunde la request-uri
- [ ] Fiecare predictie e logata in fisier
- [ ] Log-urile sunt persistente dupa restart
- [ ] Testat cu 5+ predictii
- [ ] `docker-compose down` + verificat persistenta log-urilor
- [ ] Explicat in 3+ propozitii: de ce e util docker-compose?

---

### Task 5: Kubernetes - Intelegere Conceptuala (OPTIONAL)
**Dificultate / Difficulty: Exploratory | Timp / Time: 30 min lectura + 30 min optional practic**

**RO:** Acest task este OPTIONAL si de intelegere conceptuala.
Kubernetes (K8s) e un tool complex care in industrie se invata
in luni, nu ore. Scopul aici este sa intelegi CE face si DE CE
exista, nu sa devii expert.

**EN:** This task is OPTIONAL and conceptual. Kubernetes (K8s) is
a complex tool that takes months to learn in industry, not hours.
The goal here is to understand WHAT it does and WHY it exists,
not to become an expert.

**Partea 1: Citeste si raspunde (15 min)**

Raspunde la aceste intrebari in propriile cuvinte (3-5 propozitii fiecare):

1. **Ce problema rezolva Kubernetes pe care Docker singur nu o rezolva?**
   - Hint: gandeste-te la ce se intampla cand ai 1000 de utilizatori simultani

2. **Ce e un "Pod" in Kubernetes?**
   - Hint: e cel mai mic "lucru" pe care Kubernetes il gestioneaza

3. **Ce inseamna "scaling" si de ce e important?**
   - Hint: Black Friday vs o zi normala

4. **De ce NU ai nevoie de Kubernetes pentru proiectul tau acum?**
   - Hint: cand ai <10000 utilizatori lunar, Docker e suficient

**Partea 2: Minikube OPTIONAL (30 min)**

Daca vrei sa experimentezi Kubernetes local:

```bash
# 1. Instaleaza minikube (https://minikube.sigs.k8s.io/docs/start/)
minikube start

# 2. Creeaza un fisier deployment.yaml
```

`deployment.yaml` (starter):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 2                    # 2 copii ale API-ului
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: my-ml-api:latest  # Imaginea Docker din Task 2
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  type: NodePort
  selector:
    app: ml-api
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30080
```

```bash
# 3. Aplica configuratia
kubectl apply -f deployment.yaml

# 4. Verifica
kubectl get pods          # Vezi cele 2 replici
kubectl get services      # Vezi serviciul

# 5. Acceseaza
minikube service ml-api-service --url

# 6. Scaleaza (asta e magia!)
kubectl scale deployment ml-api --replicas=5
kubectl get pods          # Acum sunt 5 copii!

# 7. Curata
minikube stop
minikube delete
```

**Evaluare / Evaluation:**
- [ ] Raspunsuri la cele 4 intrebari (Partea 1)
- [ ] Fiecare raspuns are 3-5 propozitii in cuvinte proprii
- [ ] (OPTIONAL) Minikube instalat si pornit
- [ ] (OPTIONAL) Deployment creat cu `kubectl apply`
- [ ] (OPTIONAL) Screenshot cu `kubectl get pods`
- [ ] (OPTIONAL) Scaling testat (2 → 5 replici)

---

**Resurse utile:**
- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Postman: https://www.postman.com/downloads/
- FastAPI docs: https://fastapi.tiangolo.com/
- Flask docs: https://flask.palletsprojects.com/
