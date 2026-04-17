import sqlite3

def migrate():
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    
    # Try adding assignment_file to assignments
    try:
        cur.execute("ALTER TABLE assignments ADD COLUMN assignment_file TEXT;")
        print("Added assignment_file to assignments schema.")
    except sqlite3.OperationalError as e:
        print(f"Schema potentially already updated: {e}")

    # Try adding submission_file to submissions
    try:
        cur.execute("ALTER TABLE submissions ADD COLUMN submission_file TEXT;")
        print("Added submission_file to submissions schema.")
    except sqlite3.OperationalError as e:
        print(f"Schema potentially already updated: {e}")

    con.commit()
    con.close()

if __name__ == "__main__":
    migrate()
