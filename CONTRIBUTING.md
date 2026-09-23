# Team contribution guide

1. Create an issue with the workstream, objective, inputs, output and acceptance criteria.
2. Agree an owner; avoid editing the same task in parallel without coordination.
3. Create a branch: `git switch -c cleaning/your-task` (or `import/`, `model/`, `results/`).
4. Keep changes small. Use the existing access library and record data versions.
5. Run `python -m pip install pytest` then `python -m pytest -q`.
6. Push your branch and open a pull request against `main`. Ask a teammate to review.
7. Merge after review and relevant checks; link the issue and update its status.

Do not commit access keys, passwords, `.env` files, local datasets, raw vendor files,
model binaries or row-level notebook outputs. Scrub notebook output before committing.
Do not publish or redistribute licensed source data without permission.

Ownership: Mike coordinates repository and AWS access. Isha's presentation provides
the research scope. Data, cleaning, modelling and validation owners remain to be
assigned by the team; no responsibilities have been assigned to classmates automatically.
