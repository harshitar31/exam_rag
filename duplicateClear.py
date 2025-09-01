import sqlite3

conn = sqlite3.connect("questions.db")
cur = conn.cursor()

cur.execute("""
    DELETE FROM questions
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM questions
        GROUP BY question_text, course_code, year
    )
""")

conn.commit()
conn.close()
