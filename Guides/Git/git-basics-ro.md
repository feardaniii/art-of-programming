Acest ghid este o listă practică de onboarding pentru noțiunile de bază ale controlului versiunilor: commit-uri locale, branch-uri, push, pull și pull request-uri.

## Obiective

Membrii sunt pregătiți când pot:

- Să explice diferența dintre Git și GitHub.
- Să creeze un repository local și să facă un commit.
- Să verifice starea repository-ului cu `git status` și să inspecteze istoricul cu `git log`.
- Să creeze și să schimbe branch-uri.
- Să trimită un branch pe GitHub cu push.
- Să aducă actualizări dintr-un repository remote cu pull.
- Să deschidă un pull request, să ceară review, să răspundă la feedback și să facă merge.

## Idei de bază

- Git este unealta de control al versiunilor instalată pe calculatorul tău.
- GitHub este o platformă de găzduire și colaborare pentru repository-uri Git.
- Un repository este un proiect urmărit de Git.
- Un commit este un snapshot salvat, împreună cu un mesaj.
- Un branch este o linie independentă de lucru.
- Un remote este o copie găzduită a repository-ului, de obicei numită `origin`.
- Un pull request este un flux de review pe GitHub pentru integrarea unui branch în altul. Nu este o comandă nativă Git.

## Flux zilnic de lucru

Începe prin a actualiza branch-ul local `main`:

```bash
git switch main
git pull origin main
```

Creează un branch pentru o singură sarcină:

```bash
git switch -c feature/short-description
```

Verifică ce s-a modificat:

```bash
git status
git diff
```

Pune modificările în staging și fă commit:

```bash
git add path/to/file
git commit -m "Descrie modificarea"
```

Trimite branch-ul cu push:

```bash
git push -u origin feature/short-description
```

Deschide un pull request pe GitHub din `feature/short-description` către `main`.

După ce pull request-ul este integrat prin merge, actualizează branch-ul local `main`:

```bash
git switch main
git pull origin main
git branch -d feature/short-description
```

## Comenzi esențiale

| Sarcină | Comandă |
| --- | --- |
| Inițializează un repository | `git init` |
| Clonează un repository existent | `git clone <url>` |
| Vezi starea curentă | `git status` |
| Vezi modificările neadăugate în staging | `git diff` |
| Adaugă modificări în staging | `git add <file>` |
| Fă commit pentru modificările din staging | `git commit -m "Mesaj"` |
| Afișează istoricul commit-urilor | `git log --oneline --graph --decorate` |
| Creează și schimbă branch-ul | `git switch -c <branch>` |
| Schimbă branch-ul | `git switch <branch>` |
| Listează branch-urile | `git branch` |
| Trimite branch-ul curent cu push | `git push -u origin <branch>` |
| Adu actualizări remote cu pull | `git pull origin <branch>` |
| Vezi remote-urile | `git remote -v` |

## Push, Pull, Branch, Pull Request

### Push

`git push` încarcă commit-urile locale într-un repository remote.

Folosește-l când commit-urile există local și trebuie salvate sau partajate:

```bash
git push -u origin my-branch
```

Opțiunea `-u` salvează branch-ul upstream, astfel încât viitoarele push-uri pot fi, de obicei, doar:

```bash
git push
```

### Pull

`git pull` descarcă modificările remote și le integrează în branch-ul curent.

Folosește-l înainte să începi lucrul și ori de câte ori branch-ul tău are nevoie de actualizări din remote:

```bash
git pull origin main
```

Dacă Git raportează conflicte, deschide fișierele aflate în conflict, rezolvă secțiunile marcate, apoi rulează:

```bash
git add <resolved-file>
git commit
```

### Branch

Folosește branch-uri pentru a ține fiecare sarcină separată de lucrul finalizat.

Numele bune de branch-uri sunt scurte și descriptive:

```text
feature/add-login-form
fix/navbar-mobile-spacing
docs/git-basics-guide
```

Creează un branch pornind de la `main` actualizat:

```bash
git switch main
git pull origin main
git switch -c docs/git-basics-guide
```

### Pull Request

Un pull request cere echipei să revizuiască și să integreze prin merge un branch în altul.

Înainte să deschizi un pull request:

- Confirmă că branch-ul conține doar modificarea intenționată.
- Rulează verificările sau testele relevante.
- Trimite branch-ul pe GitHub cu push.
- Scrie un titlu clar și o descriere scurtă.
- Leagă task-ul sau issue-ul asociat, când există.

În timpul review-ului:

- Răspunde direct la întrebările reviewerilor.
- Trimite commit-uri de follow-up pe același branch.
- Așteaptă verificările și aprobările înainte de merge.

## Laborator practic

Fiecare membru ar trebui să completeze acest exercițiu într-un repository de practică.

1. Clonează repository-ul.
2. Creează un branch numit `docs/<your-name>-git-practice`.
3. Adaugă un fișier markdown cu un paragraf despre ce este Git.
4. Rulează `git status`.
5. Pune fișierul în staging și fă commit.
6. Trimite branch-ul cu push.
7. Deschide un pull request către `main`.
8. Cere review de la un coechipier.
9. Fă o modificare cerută și trimite-o din nou cu push.
10. Fă merge pentru pull request după aprobare.
11. Adu local branch-ul `main` actualizat cu pull.

## Checklist de pregătire pentru membri

- [ ] Poate defini Git, GitHub, repository, commit, branch, remote și pull request.
- [ ] Poate rula `git status` înainte și după adăugarea modificărilor în staging.
- [ ] Poate face un commit curat, cu un mesaj util.
- [ ] Poate crea un branch pentru o sarcină.
- [ ] Poate trimite un branch pe GitHub cu push.
- [ ] Poate aduce actualizări din `main` cu pull.
- [ ] Poate deschide un pull request cu titlu și descriere utile.
- [ ] Poate actualiza un pull request după feedback de review.
- [ ] Poate rezolva un conflict simplu de merge cu îndrumare.

## Reguli de echipă

- Păstrează `main` deployable sau gata de prezentare.
- Folosește un branch separat pentru fiecare sarcină.
- Adu ultimul `main` cu pull înainte să începi lucru nou.
- Nu comite secrete, parole, chei API, medii virtuale sau foldere generate de build.
- Preferă pull request-uri mici, ușor de revizuit.
- Scrie mesajele de commit la imperativ, de exemplu `Add Git basics guide`.

## Depanare

Verifică pe ce branch ești:

```bash
git branch --show-current
```

Scoate din staging fără să ștergi modificările:

```bash
git restore --staged <file>
```

Aruncă modificările locale dintr-un singur fișier doar când ești sigur:

```bash
git restore <file>
```

Vezi către ce indică `origin`:

```bash
git remote -v
```

Adu informații despre branch-urile remote fără merge:

```bash
git fetch origin
```

## Resurse recomandate

Referințe oficiale:

- [Cartea Pro Git, ediția online oficială](https://git-scm.com/book/en/v2.html) - cea mai bună referință Git completă.
- [Documentația Git pentru pull](https://git-scm.com/docs/git-pull) - referința curentă pentru comandă; cea mai recentă pagină de documentație observată la review era pentru Git 2.53.0, actualizată pe 2026-02-02.
- [GitHub Docs: Start your journey](https://docs.github.com/en/get-started/start-your-journey) - parcursul GitHub pentru începători.
- [GitHub Docs: Pull requests](https://docs.github.com/pull-requests) - documentația oficială pentru fluxul de pull request.
- [GitHub Blog: Beginner's guide to creating a pull request](https://github.blog/developer-skills/github/beginners-guide-to-github-creating-a-pull-request/) - walkthrough vizual pentru începători despre PR-uri.

Videouri YouTube:

- [Corey Schafer: Git Tutorial for Beginners: Command-Line Fundamentals](https://www.youtube.com/watch?v=HVsySz-h9r4) - introducere clasică în linia de comandă, publicată pe 2015-08-03, încă utilă pentru fundamente.
- [GitHub: How to create a pull request in 4 min](https://www.youtube.com/watch?v=nCKdihvneS0) - walkthrough scurt oficial GitHub pentru PR-uri, publicat pe 2024-08-12.
- [Traversy Media: Git & GitHub Crash Course 2025](https://www.youtube.com/watch?v=vA5TTz6BXhY) - crash course actual pentru începători, publicat pe 2025-01-13.
- [freeCodeCamp: Git & GitHub Crash Course for Beginners [2026]](https://www.youtube.com/watch?v=mAFoROnOfHs) - curs recent mai lung pentru începători, publicat pe 2025-12-04.
- [ByteByteGo: How Git Works: Explained in 4 Minutes](https://www.youtube.com/watch?v=e9lnsKot_SQ) - explicație conceptuală rapidă.

## Sesiune recomandată de 60 de minute

| Timp | Activitate |
| --- | --- |
| 0-10 min | Explică Git vs GitHub, commit-uri, branch-uri și remote-uri. |
| 10-25 min | Demonstrează `status`, `add`, `commit`, `log` și `diff`. |
| 25-40 min | Demonstrează crearea unui branch, push și crearea unui pull request. |
| 40-55 min | Membrii completează laboratorul practic. |
| 55-60 min | Revizuiți checklist-ul de pregătire și răspundeți la blocaje. |

## Definiția finalizării

Acest element din checklist este complet când fiecare membru a deschis și a integrat prin merge un pull request de practică, poate explica fluxul de bază cu propriile cuvinte și știe când să ceară ajutor înainte să rescrie istoricul sau să folosească force-push.
