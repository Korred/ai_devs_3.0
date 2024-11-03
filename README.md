# AI Devs 3 with Python

This repository contains Python solutions for all AI Devs 3 related tasks.

## Important URLs

| Type    | URL |
| -------- | ------- |
| Training Ground | https://poligon.aidevs.pl/     |
| Pre-Lessons (C00)  | https://bravecourses.circle.so/c/prework-ai3/    |



## Project Setup

Ensure a ``.env`` file is filled out with all necessary data (see ``.env.local``)

### Without Docker
1. Setup a Python 3.12 environment any way you like (vanilla Python, venv, conda)
2. Use the ``requirements.txt`` to install all required dependencies/modules

### With Docker

**TODO :(**

## Running tasks

To run a specific task use the following command from the ``ai_devs`` folder:

```
python -m tasks.{course_lesson_id}.{task_name} e.g. python -m tasks.C00LXX.poligon
```


## TODOS

- Improve this file
- Add a Docker setup (once PostgresSQL and/or a vector database is added to the project)
- Switch from ``conda`` to ``uv`` (https://docs.astral.sh/uv/) for package/project management