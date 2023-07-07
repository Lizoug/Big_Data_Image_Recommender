import sqlite3
from sqlite3 import Error


def get_paths_from_db(connection, ids):
    paths = []
    for id in ids:
        df = pd.read_sql(f'SELECT * FROM paths WHERE ID = {id}', connection)
        paths.append(df['Path'].values[0])
    return paths

def save_to_sqlite(df, db_name, table_name):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    
# path_df = pd.read_pickle('ID_path_3.pkl')
# save_to_sqlite(path_df, 'paths_db_3.sqlite', 'paths')
