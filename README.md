# Study Day Assistant Agent

## Description
**Study Day Assistant** is a practical AI agent built with **LangGraph** and **ChatGroq** (`openai/gpt-oss-20b`). It helps students manage their study tasks, check real-time weather in Tbilisi via the Open-Meteo API, and search for books via the Open Library API. The agent maintains conversation memory and requires explicit user confirmation before executing destructive actions (`delete_task`).

## Features & Tools
- `add_task`: Adds a new task to the study list.
- `list_tasks`: Displays all current tasks.
- `delete_task`: Deletes a task (protected by a Human-in-the-Loop approval interrupt).
- `get_tbilisi_weather`: Fetches current temperature in Tbilisi using Open-Meteo API.
- `search_books`: Searches books by topic using Open Library API.

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
