import sqlite3

con = sqlite3.connect("recv/events.db")
con.row_factory = sqlite3.Row

print("=== EVENTS ===")
rows = con.execute("SELECT * FROM events ORDER BY rowid DESC LIMIT 5").fetchall()
for r in rows:
    print(dict(r))

print("\n=== FRAMES ===")
rows = con.execute("SELECT * FROM frames ORDER BY rowid DESC LIMIT 5").fetchall()
for r in rows:
    print(dict(r))

con.close()