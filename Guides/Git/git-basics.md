This guide is a practical onboarding checklist for version control basics: local commits, branches, pushing, pulling, and pull requests.

## Outcomes

Members are ready when they can:

- Explain the difference between Git and GitHub.
- Create a local repository and make a commit.
- Check repository state with `git status` and inspect history with `git log`.
- Create and switch branches.
- Push a branch to GitHub.
- Pull updates from a remote repository.
- Open a pull request, request review, respond to feedback, and merge.

## Core Ideas

- Git is the version control tool installed on your machine.
- GitHub is a hosting and collaboration platform for Git repositories.
- A repository is a project tracked by Git.
- A commit is a saved snapshot with a message.
- A branch is an independent line of work.
- A remote is a hosted copy of the repository, usually named `origin`.
- A pull request is a GitHub review workflow for merging one branch into another. It is not a native Git command.

## Daily Workflow

Start by updating your local `main` branch:

```bash
git switch main
git pull origin main
```

Create a branch for one task:

```bash
git switch -c feature/short-description
```

Check what changed:

```bash
git status
git diff
```

Stage and commit:

```bash
git add path/to/file
git commit -m "Describe the change"
```

Push your branch:

```bash
git push -u origin feature/short-description
```

Open a pull request on GitHub from `feature/short-description` into `main`.

After the pull request is merged, update local `main`:

```bash
git switch main
git pull origin main
git branch -d feature/short-description
```

## Essential Commands

| Task                         | Command                                |
| ---------------------------- | -------------------------------------- |
| Initialize a repository      | `git init`                             |
| Clone an existing repository | `git clone <url>`                      |
| See current state            | `git status`                           |
| See unstaged changes         | `git diff`                             |
| Stage changes                | `git add <file>`                       |
| Commit staged changes        | `git commit -m "Message"`              |
| Show commit history          | `git log --oneline --graph --decorate` |
| Create and switch branch     | `git switch -c <branch>`               |
| Switch branch                | `git switch <branch>`                  |
| List branches                | `git branch`                           |
| Push current branch          | `git push -u origin <branch>`          |
| Pull remote updates          | `git pull origin <branch>`             |
| See remotes                  | `git remote -v`                        |

## Push, Pull, Branch, Pull Request

### Push

`git push` uploads local commits to a remote repository.

Use it when your commits exist locally and need to be backed up or shared:

```bash
git push -u origin my-branch
```

The `-u` option records the upstream branch, so future pushes can usually be just:

```bash
git push
```

### Pull

`git pull` downloads remote changes and integrates them into your current branch.

Use it before starting work and whenever your branch needs updates from the remote:

```bash
git pull origin main
```

If Git reports conflicts, open the conflicted files, resolve the marked sections, then run:

```bash
git add <resolved-file>
git commit
```

### Branch

Use branches to keep one task separate from finished work.

Good branch names are short and descriptive:

```text
feature/add-login-form
fix/navbar-mobile-spacing
docs/git-basics-guide
```

Create a branch from updated `main`:

```bash
git switch main
git pull origin main
git switch -c docs/git-basics-guide
```

### Pull Request

A pull request asks the team to review and merge one branch into another.

Before opening a pull request:

- Confirm the branch contains only the intended change.
- Run the relevant checks or tests.
- Push the branch to GitHub.
- Write a clear title and short description.
- Link the related task or issue when available.

During review:

- Answer reviewer questions directly.
- Push follow-up commits to the same branch.
- Wait for checks and approvals before merging.

## Practice Lab

Each member should complete this in a practice repository.

1. Clone the repository.
2. Create a branch named `docs/<your-name>-git-practice`.
3. Add a markdown file with one paragraph about what Git is.
4. Run `git status`.
5. Stage and commit the file.
6. Push the branch.
7. Open a pull request into `main`.
8. Ask one teammate for review.
9. Make one requested edit and push again.
10. Merge the pull request after approval.
11. Pull the updated `main` branch locally.

## Member Readiness Checklist

- [ ] Can define Git, GitHub, repository, commit, branch, remote, and pull request.
- [ ] Can run `git status` before and after staging changes.
- [ ] Can make a clean commit with a useful message.
- [ ] Can create a branch for a task.
- [ ] Can push a branch to GitHub.
- [ ] Can pull updates from `main`.
- [ ] Can open a pull request with a useful title and description.
- [ ] Can update a pull request after review feedback.
- [ ] Can resolve a simple merge conflict with guidance.

## Team Rules

- Keep `main` deployable or presentation-ready.
- Use a separate branch for each task.
- Pull latest `main` before starting new work.
- Do not commit secrets, passwords, API keys, virtual environments, or generated build folders.
- Prefer small pull requests that are easy to review.
- Write commit messages in the imperative style, for example `Add Git basics guide`.

## Troubleshooting

Check the branch you are on:

```bash
git branch --show-current
```

Undo staging without deleting edits:

```bash
git restore --staged <file>
```

Discard local edits to one file only when you are sure:

```bash
git restore <file>
```

See where `origin` points:

```bash
git remote -v
```

Fetch remote branch information without merging:

```bash
git fetch origin
```

## Recommended Resources

Official references:

- [Pro Git book, official online edition](https://git-scm.com/book/en/v2.html) - best complete Git reference.
- [Git pull documentation](https://git-scm.com/docs/git-pull) - current command reference; the latest docs page observed during review was for Git 2.53.0, updated 2026-02-02.
- [GitHub Docs: Start your journey](https://docs.github.com/en/get-started/start-your-journey) - GitHub's beginner path.
- [GitHub Docs: Pull requests](https://docs.github.com/pull-requests) - official pull request workflow documentation.
- [GitHub Blog: Beginner's guide to creating a pull request](https://github.blog/developer-skills/github/beginners-guide-to-github-creating-a-pull-request/) - visual beginner PR walkthrough.

YouTube videos:

- [Corey Schafer: Git Tutorial for Beginners: Command-Line Fundamentals](https://www.youtube.com/watch?v=HVsySz-h9r4) - classic command-line intro, published 2015-08-03, still useful for fundamentals.
- [GitHub: How to create a pull request in 4 min](https://www.youtube.com/watch?v=nCKdihvneS0) - short official GitHub PR walkthrough, published 2024-08-12.
- [Traversy Media: Git & GitHub Crash Course 2025](https://www.youtube.com/watch?v=vA5TTz6BXhY) - current beginner crash course, published 2025-01-13.
- [freeCodeCamp: Git & GitHub Crash Course for Beginners [2026]](https://www.youtube.com/watch?v=mAFoROnOfHs) - recent longer beginner course, published 2025-12-04.
- [ByteByteGo: How Git Works: Explained in 4 Minutes](https://www.youtube.com/watch?v=e9lnsKot_SQ) - quick conceptual explanation.

## Suggested 60-Minute Session

| Time      | Activity                                               |
| --------- | ------------------------------------------------------ |
| 0-10 min  | Explain Git vs GitHub, commits, branches, and remotes. |
| 10-25 min | Demo `status`, `add`, `commit`, `log`, and `diff`.     |
| 25-40 min | Demo branch creation, push, and pull request creation. |
| 40-55 min | Members complete the practice lab.                     |
| 55-60 min | Review readiness checklist and answer blockers.        |

## Definition of Done

This checklist item is complete when every member has opened and merged a practice pull request, can explain the basic workflow in their own words, and knows when to ask for help before rewriting history or force-pushing.
