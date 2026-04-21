# Entry Task 01: Foundation Before Scale

This is the first onboarding task for the 2-month internship.

Its purpose is simple:

1. make sure every student can access, clone, pull, branch, push, and open a PR against the repo
2. build a small but real technical seed from which the larger internship project can grow
3. study the repo and understand the development space in which the team will operate

This task is split into 2 parts across the first 2 weeks.

## Why This Task Exists

We are not starting from complexity.

We are starting from foundation:

- local environment
- Python execution
- virtual environments
- Git workflow
- API calls to an LLM provider
- local file persistence
- SQLite
- clean thinking about structure, modeling, and growth

The goal is to build a small living system that already contains the DNA of a larger one.

## Required Reading Before Starting

Read these first:

- [Root README](../README.MD)
- [Internship Project Overview](./Project_overview.md)
- [OOP Special Homework](../06_oop/Special_Task/oop_homework_english.md)

Why these 3:

- `README.MD` explains the structure, philosophy, and contribution flow of this repository.
- `Project_overview.md` shows how a real AI system is broken into stages, checkpoints, and outputs.
- `oop_homework_english.md` trains the exact mental skill needed here: seeing a real system as classes, entities, relations, and behavior.

## Step 0: Repo Access Check

Before building anything, make sure you can actually work inside the team environment.

Each student must:

1. fork or clone the repository
2. pull the latest changes
3. create a branch
4. make a tiny test contribution if requested by the mentor
5. open a pull request

This is not optional. If Git is unstable, everything above it becomes unstable.
Your first PR is the practical proof that you can enter the team workflow correctly.

## Part I: Technical Seed Build

**Deadline:** End of Week 1

### Objective

Create a small Python application that combines:

- an LLM API call
- session memory stored locally
- file-based persistence
- a SQLite database
- a simple user loop

This is the seed from which the internship tree can grow.

### Pre-Requisites

You must be comfortable enough to set up and use:

- Python interpreter(s)
- virtual environments (`venv` is enough)
- Git basics: `clone`, `pull`, `branch`, `push`, `pull request`
- a simple API call to one LLM provider:
  - OpenAI
  - Google
  - Anthropic
- local file persistence with `.txt` or `.md`
- SQLite basics

If any of these are still foggy, that is your current level of work.
Stay there until it becomes clear.

## Part I Minimum Build

Build a local application in Python with the following behavior:

1. user can talk to the LLM from a terminal app or a simple UI
2. each run of the program creates a new session
3. each session stores the conversation to a local `.txt` or `.md` file
4. on each new LLM call inside that session, the app sends the previous conversation as raw pre-context
5. the project includes a pre-created SQLite database about a restaurant inventory
6. the user can query the database through a simple loop-based menu
7. the user can choose between:
   - talking to the LLM
   - searching the database
   - exiting the application

Terminal is fully acceptable.
PySide / PyQt / another UI is optional.

### Suggested Technical Shape

You do not have to copy this exactly, but the structure should be similarly clean:

```text
project/
├── main.py
├── llm_client.py
├── memory.py
├── db.py
├── data/
│   └── inventory.db
├── conversations/
│   └── session_001.txt
└── README.md
```

### Required Behaviors

Your project should demonstrate all of the following:

- creation and activation of a virtual environment
- one clear script or entry point to run the app
- one LLM provider integration working locally
- API key loading through environment variables
- session-based conversation memory saved to disk
- SQLite database created ahead of time or initialized on first run
- at least basic inventory search functionality
- clean separation between:
  - LLM logic
  - persistence logic
  - database logic
  - application loop

### Suggested Database Model

At minimum, your SQLite database should support a realistic inventory table.

Example fields:

- `id`
- `product_name`
- `category`
- `brand`
- `supplier`
- `quantity_in_stock`
- `unit_price`
- `last_updated`

You may extend this with sales tables, partner tables, purchase history, or monthly reports.

## Part I Acceptance Criteria

Your submission passes the minimum bar if:

1. the app runs locally
2. the app can make at least one successful LLM request
3. the conversation is stored locally per session
4. the previous conversation is reused as context during the session
5. the SQLite database exists and can be queried
6. the code is organized clearly enough that another student can follow it
7. the repository link to your work is submitted

## Advanced Extension

If you finish the minimum build, extend it.

### Advanced Goal

Find a way to let the LLM know about your database and perform database operations such as:

- `SELECT`
- `UPDATE`
- `DELETE`

This can be done through:

- function/tool calling
- a controlled command layer
- prompt-to-SQL with strict validation
- a rule-based interpreter that converts limited user intents into safe SQL

Do not allow unrestricted dangerous execution.
Control the allowed operations.

### Advanced Scenario

Your system should move toward this kind of request:

> "We sold 100 electronics today: 15 TVs, 50 fridges, 35 washing machines, and we sold out all microwaves. Update stock, give me monthly dollar statistics, and plan the next bulk buy for electronics from our Samsung partner."

This is the point of the exercise.
The user request defines system complexity.
That means your data model matters.

Exactly as in OOP, the quality of the model determines the power of the solution.

### Advanced Result Direction

Think of the advanced version as an early form of:

**AI-Assisted Retail Shop**

The stronger the modeling, the stronger the system.

## Part II: Repo Study and Team Presentation

**Deadline:** End of Week 2

### Objective

Study this repository and prepare a presentation that explains:

- what this repo is
- what kinds of systems live inside it
- where our internship work will happen
- how development happens in practice
- how code work connects to team/project management in Notion

This is not a surface-level tour.
The goal is to build orientation.

### What You Must Study

At minimum, cover:

1. the root [README](../README.MD)
2. the internship [Project Overview](./Project_overview.md)
3. the main repo structure and module layout
4. the contribution flow described in the README
5. the relationship between:
   - scripts
   - tasks
   - student submissions
   - internship work
   - project management tracking

### Your Presentation Must Answer

Your presentation should clearly answer these questions:

1. What is the mission of this repository?
2. How is learning structured across modules?
3. Why are scripts mostly self-contained?
4. Where do students submit work?
5. How does a change move from local code to branch to PR to merge?
6. Where will our team likely work during the internship?
7. How should development progress be mirrored in Notion?
8. Which folders or files are most relevant to the internship effort?

### Notion Sync Expectation

For this internship, code work and project management should stay connected.

At minimum, students should understand this flow:

1. task appears in Notion
2. task is claimed and moved to in progress
3. work happens on a local branch
4. PR is opened
5. PR link and status are reflected back in Notion
6. after review and merge, Notion is updated to done

Your Week 2 presentation should show that you understand this loop.

### Recommended Presentation Structure

You may present in slides, Markdown, PDF, or live demo format.

Suggested structure:

1. repo mission and teaching philosophy
2. high-level folder map
3. internship-relevant files and directories
4. contribution workflow
5. how coding work syncs with Notion planning
6. your conclusions: where the project can grow next

## Submission Format

Submit:

1. a Git repo link for Part I
2. a short project README for Part I
3. your Part II presentation or presentation file link
4. a short note describing:
   - what worked
   - what was difficult
   - what you still do not understand clearly

## Evaluation Criteria

You will be evaluated on:

- environment setup correctness
- Git workflow correctness
- clarity of code structure
- correct LLM API integration
- correct local persistence
- correct SQLite usage
- quality of system modeling
- quality of repo understanding
- clarity of presentation
- seriousness of execution

## Final Thought

There are 2 skills in the Art of Programming that can completely change a student's trajectory.

### I. Humbleness and respect for the craft

You must go exactly to your real level.

If Python and virtual environments are unclear, that is where your work is.
If Git pull, branches, and PRs are unclear, that is where your work is.
Anything built above unstable ground will remain unstable.

> "Learn to regress, deserve to progress."

### II. Sincere dedication in doing

Prepare your environment.
Prepare your desk.
Prepare your folders, notes, editor, shortcuts, and habits.
Make the act of working clean and repeatable.

Any skilled builder knows where the tools are.
Programming is not different.

In a world that rushes toward fast results, these two skills remain hidden:

- humility to stand at the true current level
- dedication to build with care

Students who develop both do not only progress faster.
They progress on stable ground.
