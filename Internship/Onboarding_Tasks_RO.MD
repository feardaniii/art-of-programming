<a name="top"></a>

<h1 align="center">Task-ul de Intrare în Internship: Punerea Bazelor</h1>

<p align="center">
  <em>Bun venit la internship-ul de 2 luni! Ca parte a cohortei noastre de 40 de studenți, acest document de onboarding va servi drept task de intrare pentru a înțelege și simți Sistemul în care operăm.</em>
</p>

---

## Partea 1: Familiarizarea (Repository & Contribuitori)
- Începeți prin a explora repository-ul și a citi [`README.MD`](../README.MD) principal.
- Asigurați-vă că aveți acces la repository-ul Internship-ului și că sunteți adăugat ca și contributor.

---

## Partea 2: Tema de Casă de Două Săptămâni
Tema de onboarding este împărțită pe două săptămâni, concepută pentru a construi atât abilități tehnice, cât și sinergia echipei.

### Săptămâna 1: Fundația Tehnică (Proiectul "Sămânță")
Vom învăța să creăm mica sămânță din care proiectul nostru de Internship va crește într-un copac. Veți construi un sistem fundamental care combină Python, LLM-uri și baze de date locale.

#### Pre-rechizite (Referințe Săptămâna 1)
Înainte de a pune bazele, asigurați-vă că sunteți familiarizat cu:
- **Python:** Interpretator(oare) Python instalat(e) și configurat(e).
- **Managementul Mediului:** Managementul mediilor virtuale (`venv`).
- **Git:** Bazele controlului versiunilor (`push`, `pull`, `branch`, `pull request`).
- **Apeluri API LLM:** Scrierea unui script simplu de 10 linii pentru a invoca un apel API de pe mașina locală către un furnizor LLM (precum Google, OpenAI sau Anthropic). Manipularea cheilor API (API Keys).
- **Baze de Date:** Concepte simple de baze de date folosind fișiere locale (`.md`/`.txt`) și SQLite.

#### Punerea Bazelor
Acum haideți să punem bazele - Sistemul fundamental din care va crește cel complex.

**Descrierea Task-ului:**
Puneți totul împreună într-o aplicație Python:
1. **Integrare LLM:** Un script Python pentru a apela un LLM.
2. **Memoria Sesiunii:** Stocarea conversației într-un fișier `.txt/.json` pentru a menține o "memorie" per sesiune. Faceți un "raw dump" al întregii conversații ca Pre-Context la fiecare nou apel.
3. **Bază de Date Locală:** Rularea SQLite cu o bază de date pre-creată despre inventarul unui magazin de retail (ex. Carrefour România Cluj-Vivo).
4. **Interfața cu Utilizatorul:** Crearea unui "Meniu" bazat pe o buclă (loop) din care utilizatorul poate fie să Căute/Interogheze baza de date, fie să vorbească cu LLM-ul. Acest lucru poate fi făcut în Terminal sau folosind un framework UI (precum PySide, PyQt, etc.).

**Criterii de Acceptare**
- [ ] Scriptul Python rulează cu succes într-un `venv` izolat.
- [ ] Aplicația se conectează la un furnizor LLM prin API.
- [ ] Istoricul conversației este salvat într-un fișier `.txt` și încărcat ca și context pentru următorul apel.
- [ ] Baza de date SQLite există, având date de bază despre inventarul de Retail.
- [ ] O buclă/meniu interactiv permite utilizatorului să interogheze baza de date sau să discute cu AI-ul.

#### Provocare Avansată - Magazin de Retail Asistat de AI
**Scop:** Găsiți o modalitate de a informa LLM-ul despre baza voastră de date ȘI nu numai - o modalitate prin care acesta să poată efectua operațiuni pe baza de date (SELECT, UPDATE, DELETE) în mod autonom, bazat pe limbaj natural.

**Criterii de Acceptare Avansate:**
- LLM-ul poate interpreta un prompt și poate acționa ca un magazin de Retail Asistat de AI.
- **Exemplu de Prompt de la Utilizator:** _“Am vândut 100 de electronice astăzi - 15 TV-uri, 50 de frigidere, 35 de mașini de spălat - și am epuizat TOATE cuptoarele cu microunde. Actualizează stocul, dă-mi statisticile lunare în $ și planifică următoarea achiziție en-gros de electronice de la partenerul nostru Samsung.”_
- **Observație:** Observați cum cererea utilizatorului (Specificațiile) pictează complexitatea și conexiunile din baza de date pe care o creați (Modelarea Datelor). Acest lucru este exact similar cu Tema noastră de OOP: [Task-ul Special OOP](../06_oop/Special_Task/oop_homework_english.md)

---

### Săptămâna 2: Studiul Repository-ului & Hand-off (Proiect de Echipă)
*(Notă: Această fază nu poate fi finalizată cu succes fără fundația solidă pusă în Săptămâna 1!)*

Pentru Săptămâna 2, cohorta va fi împărțită în **3 Echipe**. Veți lucra, studia, discuta și face brainstorming împreună. 

**Misiunea Echipei Voastre:**
Studiați repository-ul și pregătiți o prezentare care să acopere:
- Ce se întâmplă în proiect?
- Unde se va desfășura munca noastră ca echipă.
- Cum funcționează codul (fluxul de dezvoltare).
- Cum se conectează la managementul proiectului din Notion pentru o sincronizare continuă.

**Linii Directoare pentru Prezentare:**
- Fiecare dintre cele 3 echipe va avea **15 minute** pentru a prezenta.
- Puteți (opțional) să folosiți mai multe persoane din echipă în timpul prezentării pentru a acoperi diferite aspecte ale repository-ului și ale fluxului de lucru.

---

## Gând Final: Arta Programării
Există 2 abilități în Arta Programării care pot schimba întreaga Experiență & Performanță a Studentului:

### I. Smerenia și Respectul pentru Meșteșug

Găsirea smereniei de a merge **exact la propriul nivel curent pentru a învăța**. 

Dacă mediile virtuale și Python sunt în ceață, acela este locul studentului pentru o vreme. 

Dacă Git și sincronizarea și ramificarea (branching) locală a repository-ului reprezintă o provocare, exact acolo trebuie să fie studentul și nicăieri altundeva mai sus în ierarhia tehnică. Orice este mai sus va fi construit pe teren instabil.

> _“Învață să Regresezi, Merită să progresezi.”_ 
> 
> Un concept care pune de asemenea baza pentru Recuperarea & Performanța în biomecanică.

### II. Dedicare Sinceră și Efort

Angajarea în muncă cu **dedicare** sinceră și **efort**. 

Găsirea timpului disponibil, găsirea locului de studiu și crearea unui mediu confortabil astfel încât să fie o adevărată plăcere statul la calculator. 

Asigurarea că biroul, mouse-ul și tastatura sunt curate. Existența apei alături, și pregătirea întregii stații de lucru așa cum s-ar pregăti pentru un _oaspete drag_.

Găsirea unui mod personal de lucru: IDE-ul preferat, hotkey-urile și scurtăturile prin PC, notițele salvate și blueprint-urile tehnice, bookmark-urile, directoarele de cod aranjate, imaginile, folderele și fișierele. 

Setarea lor în așa fel încât, dacă este nevoie de ceva, studentul să știe exact unde să caute și cum să ajungă acolo, la fel cum ar face-o orice Tâmplar sau Inginer iscusit.

Într-o lume care este într-o grabă continuă și în care Mintea este pre-setată de Evoluție către **conservarea optimă a energiei** (similar cu o rețea neuronală artificială care rămâne blocată pe un *maxim local*), căutând astfel să îndeplinească sarcinile rapid și cu succes, ambele aceste abilități sunt adesea ‘ascunse’. 

Sigur, studentul poate progresa, dar fără acestea, rămâne departe de adevăratul său **Potențial**.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/LLM-Integration-FF6F00?style=for-the-badge&logo=openai&logoColor=white" alt="LLM">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/modules-30-blue?style=flat-square" alt="Modules">
  <img src="https://img.shields.io/badge/scripts-120+-green?style=flat-square" alt="Scripts">
  <img src="https://img.shields.io/badge/visualizations-269-orange?style=flat-square" alt="Visualizations">
</p>

<p align="center">
  <sub>Construit pentru mintea curioasă. <br></br>Călătoria începe.</sub>
</p>

[&uarr; Înapoi sus](#top)