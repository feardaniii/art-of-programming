# Probleme identificate

## 1. Derapaj în Compose pentru Redis Stack / RediSearch

`docker-compose.dev.yml` nu mai era aliniat cu cerințele actuale ale aplicației; rulările locale depindeau de Redis Stack / RediSearch, nu de Redis simplu, ceea ce făcea ca pornirea locală să eșueze înainte ca aplicația să poată rula corect.

Căi: `docker-compose.dev.yml`

Soluția folosită: serviciul din Compose a fost schimbat din `redis:7-alpine` în Redis Stack, iar comanda a fost actualizată la `redis-stack-server`.

Evidență:

![Eroare Redis Stack FT._LIST](screenshots/redis-stack-ft-list-error.png)

## 2. Importul opțional pentru checkpoint Postgres putea opri pornirea

Checkpointing-ul Postgres este opțional în configurația actuală, dar pornirea putea totuși să eșueze complet când dependența lipsea, ceea ce însemna că serviciul nu pornea deși acea funcționalitate nu era necesară.

Căi: `src/server.py`, `src/graphs/course_generation/persistence.py`

Soluția folosită: importul opțional pentru checkpointer-ul Postgres a fost protejat, astfel încât lipsa `langgraph-checkpoint-postgres` să degradeze comportamentul la checkpointing istoric dezactivat, fără să oprească pornirea aplicației.

Evidență:

![Eroare import Postgres checkpointer](screenshots/postgres-checkpointer-import-error.png)

## 3. Nepotrivire între logger factory și configurația structlog

Configurația structlog se baza pe semantica stdlib logging, dar logger factory-ul configurat nu era compatibil, ceea ce provoca un crash imediat la pornire în loc de logging normal.

Căi: `src/server.py`

Soluția folosită: `structlog.PrintLoggerFactory()` a fost înlocuit cu `structlog.stdlib.LoggerFactory()`.

Evidență:

![Crash la pornire din cauza Structlog PrintLogger](screenshots/structlog-printlogger-startup-crash.png)

## 4. Valorile implicite Docker pentru env indicau `localhost`

Valorile implicite din env pentru rulările containerizate erau înșelătoare, deoarece `localhost` nu se rezolvă către serviciul intenționat, ceea ce făcea ca aplicația să nu își poată accesa dependențele în rulările locale.

Căi: `.env.example`, `src/server.py`, `docs/QUICKSTART.md`, `docs/ENVIRONMENT.md`, `docs/README.md`, `docs/API_REFERENCE.md`

Soluția folosită: pentru rulările cu Compose, în `REDIS_URL` a fost folosit hostname-ul serviciului Docker în loc de `localhost`.

Evidență:

![Eroare conexiune Redis pe localhost](screenshots/redis-localhost-connection-error.png)
