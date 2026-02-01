# Concluzii

Modelul Random Forest obține performanțe mult superioare regresiei liniare datorită capacității de a modela relații neliniare. Performanța aproape perfectă este explicabilă prin faptul că datele conțin variabile care determină direct temperatura stelei.

---

## R² – Coeficientul de determinare

### Definiție

R² măsoară cât de bine explică modelul variația datelor, comparativ cu un model banal care ar prezice doar media.

### Cum se interpretează R² (foarte important)

| Valoare R² | Interpretare |
|------------|--------------|
| 1.0 | predicție perfectă |
| 0.95 | model foarte bun |
| 0.5 | explică jumătate din variație |
| 0.0 | la fel de prost ca media |
| < 0 | mai prost decât media |

---

## De ce Random Forest bate Linear Regression

Temperatura este matematic „codificată" în Luminosity + Radius.

**Random Forest:**
- nu știe fizică
- dar învață perfect această relație
- și o face aproape fără eroare

**Linear Regression:**
- vede doar o aproximare liniară
- pierde termenii neliniari (puterea a 4-a!)

---

## De ce rezultatul RF este „suspect de bun" (dar explicabil)

Nu este un bug.
Nu este o eroare de cod.
Este un caz clasic de **leakage fizic / determinism**.

Ce se întâmplă de fapt:
- modelul NU generalizează „astronomie"
- modelul reconstruiește o formulă implicită
- train și test provin din același mecanism de generare

De aceea:
- R² → 1.0
- erori → foarte mici
