import sqlite3
import pandas as pd
import glob

def conectar_sqlite():
    """ create a database connection to an SQLite database """
    conn = None
    filename = "aneel_ped.db"
    try:
        conn = sqlite3.connect(filename)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            return conn
            
def write_table(conn, file, table_name):
    # read file
    df = pd.read_csv(file, sep=',')
    # write the data to a sqlite table
    df.to_sql(table_name, conn, if_exists='replace', index = False)

def get_files():
    extension = 'csv'
    result = glob.glob(f'data/*.{extension}')
    print(result)
    return(result)

def save_tables_in_db():
    conn = conectar_sqlite()
    for i in get_files():
        write_table(conn=conn, file=i, table_name=i.replace("data/", "").replace(".csv", ""))
    return "Tabelas salvas"
    
