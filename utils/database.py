# -----------------------------------------------------------------------------
# utils/database.py - SQLite Database for Project Management
# -----------------------------------------------------------------------------

import sqlite3
import os
import shutil

# Define the database file path
DB_PATH = 'site_monitor.db'
# Define the base directory for project data
DATA_DIR = 'data'

def init_db():
    """Initializes the SQLite database and the data directory structure."""
    print("Initializing database...")
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Create projects table
        c.execute('''CREATE TABLE IF NOT EXISTS projects
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT UNIQUE NOT NULL,
                      location TEXT NOT NULL)''')
        conn.commit()
        print("Database initialized successfully.")

        # Create base data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Data directory '{DATA_DIR}' ensured.")

    except Exception as e:
        print(f"Error initializing database or data directory: {e}")
    finally:
        if conn:
            conn.close()

def create_project(name, location):
    """
    Creates a new project in the database and sets up its directory structure.

    Returns:
        int: The new project ID if successful, None if project name exists.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO projects (name, location) VALUES (?, ?)", (name, location))
        project_id = c.lastrowid
        conn.commit()

        # Create project directory structure
        project_dir = os.path.join(DATA_DIR, str(project_id))
        os.makedirs(os.path.join(project_dir, 'media'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'processed_media'), exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'reports'), exist_ok=True)

        print(f"Project '{name}' created with ID {project_id}.")
        return project_id
    except sqlite3.IntegrityError:
        print(f"Error creating project: Project name '{name}' already exists.")
        return None # Project name already exists
    except Exception as e:
        print(f"Error creating project or directories: {e}")
        if conn:
             conn.rollback() # Rollback DB changes on error
        return None
    finally:
        if conn:
            conn.close()

def get_projects():
    """Retrieves all projects from the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, location FROM projects ORDER BY name")
        projects = c.fetchall() # Returns list of tuples (id, name, location)
        return [{"id": row[0], "name": row[1], "location": row[2]} for row in projects]
    except Exception as e:
        print(f"Error retrieving projects: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_project(project_id):
    """Retrieves a single project by ID."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, location FROM projects WHERE id = ?", (project_id,))
        project = c.fetchone()
        if project:
            return {"id": project[0], "name": project[1], "location": project[2]}
        return None
    except Exception as e:
        print(f"Error retrieving project {project_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def delete_project(project_id):
    """Deletes a project from the database and its associated data directory."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()

        # Delete associated directory
        project_dir = os.path.join(DATA_DIR, str(project_id))
        if os.path.exists(project_dir):
             shutil.rmtree(project_dir)
             print(f"Deleted project directory: {project_dir}")

        print(f"Project with ID {project_id} deleted.")
        return True
    except Exception as e:
        print(f"Error deleting project {project_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# Initialize the database and data directory when the module is imported
init_db()