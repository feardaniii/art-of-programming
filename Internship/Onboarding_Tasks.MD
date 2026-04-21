<a name="top"></a>

<h1 align="center">Internship Entry Task: Laying the Foundation</h1>

<p align="center">
  <em>Welcome to the 2-month internship! As part of our cohort of 40 students, this onboarding document will serve as your entry task to get a grasp and feel of the System we are operating in.</em>
</p>

---

## Part 1: Getting Familiar (Repository & Contributors)
- Start by exploring the repository and reading the main [`README.MD`](../README.MD).
- Make sure you have access to the Internship repository and are added as a contributor.

---

## Part 2: The Two-Week Homework
The onboarding homework is split into two weeks, designed to build both technical skill and team synergy.

### Week 1: Technical Foundation (The Seed Project)
We will learn to create the little seed from which our Internship project will grow into a tree. You will build a foundational system combining Python, LLMs, and local databases.

#### Pre-requisites (Week 1 References)
Before building the foundation, ensure you are comfortable with:
- **Python:** Python interpreter(s) installed and configured.
- **Environment Management:** Virtual environment (`venv`) management.
- **Git:** Version control basics (`push`, `pull`, `branch`, `pull request`).
- **LLM API Calls:** Writing a simple 10-line script to invoke an API request from your local machine to an LLM provider (like Google, OpenAI, or Anthropic). Handling of API Keys.
- **Databases:** Simple database concepts using local files (`.md`/`.txt`) and SQLite.

#### Laying the Foundation
Now let's lay the foundation - the Foundational system from where the complex one will grow.

**Task Description:**
Put it all together into a Python application:
1. **LLM Integration:** A Python script to call an LLM.
2. **Session Memory:** Store the conversation in a `.txt/.json` file to maintain a per-session "memory". Raw dump the whole conversation as Pre-Context in each new call.
3. **Local Database:** Have SQLite running with a pre-created database about a retail shop (ex. Carrefour Romania Cluj-Vivo) inventory.
4. **User Interface:** Create a loop-based "Menu" from which the user can either Query/Search the database or talk to the LLM. This can be done in the Terminal or using a UI framework (like PySide, PyQt, etc.).

**Acceptance Criteria**
- [ ] Python script successfully runs within an isolated `venv`.
- [ ] Application connects to an LLM provider via API.
- [ ] Conversation history is saved in a `.txt` file and loaded as context for the next call.
- [ ] SQLite database exists with basic Retail inventory data.
- [ ] Interactive loop/menu allows the user to query the database or chat with the AI.

#### Advanced Challenge - AI-Assisted Retail Shop
**Goal:** Find a way to let the LLM know about your database AND more - a way that it can perform database operations (SELECT, UPDATE, DELETE) autonomously based on natural language.

**Advanced Acceptance Criteria:**
- The LLM can interpret a prompt and act as an AI-Assisted Retail shop.
- **Example User Prompt:** _“We sold 100 electronics today - 15 TVs, 50 fridges, 35 washing machines - and we sold-out ALL microwaves. Update stock, give me monthly $ statistics - and plan the next bulk-buy electronics from our Samsung partner.”_
- **Notice:** Observe how the user request (Specifications) paints the complexity and connections in the database you create (Data Modelling). This is exactly similar to our OOP Homework: [OOP Special Task](../06_oop/Special_Task/oop_homework_english.md)

---

### Week 2: Repository Study & Hand-off (Team Project)
*(Note: This phase cannot be successfully completed without the solid foundation laid in Week 1!)*

For Week 2, the cohort will be divided into **3 Teams**. You will work, study, talk, and brainstorm together. 

**Your Team's Mission:**
Study the repository and prepare a presentation covering:
- What is happening in the project?
- Where our work as a team will happen.
- How the code works (development workflow).
- How it connects to our Notion project management for continuous sync.

**Presentation Guidelines:**
- Each of the 3 teams will have **15 minutes** to present.
- You can (optional) use multiple people from your team during the presentation to cover different aspects of the repository and workflow.

---

## Final Thought: The Art of Programming
There are 2 skills in the Art of Programming which can change the whole Experience & Performance of the Student:

### I. Humbleness and Respect for the Craft

Finding the humbleness to go **exactly to one's current level to learn**. 

If virtual environments and Python are foggy, that is the student's place for a while. 

If Git and getting the repo synced and branched locally is challenging, that is exactly where the student needs to be and nowhere else higher in the technical hierarchy. Anything higher will be built on unstable grounds.

> _“Learn to Regress, Deserve to progress.”_ 
> 
> A concept which lays also the foundation for biomechanics Recovery & Performance.

### II. Sincere Dedication and Effort

Engaging in the work with sincere **dedication** and **effort**. 

Finding the available time, finding the place of study, and creating a comfortable environment so that it is a true pleasure to sit at the computer. 

Ensuring the desk, mouse, and keyboard are clean. Having water nearby, and preparing the whole working station as one would prepare it for a _loving guest_.

Finding a personal way of working: the preferred IDE, the hotkeys and shortcuts around the PC, the saved notes and technical blueprints, bookmarks, arranged code directories, images, folders, and files. 

Setting it up so that if anything is needed, the student knows exactly where to look and how to get there, as would any skilled Carpenter or Engineer.

In a world which is in a continuous rush and the Mind is pre-set by Evolution towards **optimal energy conservation** (similar to an artificial neural network which remains stuck on a *local maximum*), and thus seeking to perform tasks fast and successfully, both these skills are often ‘hidden’. 

Sure, the student may be progressing, but without these, they remain far from their true **Potential**.

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
  <sub>Built for the curious mind. <br></br>The Journey begins.</sub>
</p>

[&uarr; Back to top](#top)