def create_tables():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    table = c.execute("""CREATE TABLE employee (
                full_name text NOT NULL,
                age integer,
                rating REAL NOT NULL,
                remarks text)""")
    return table


def insert_values(conn, rows):
    c = conn.cursor()
    return c.executemany("INSERT INTO players VALUES (?, ?, ?, ?)", rows)


def append_values(conn, df):
    sql = """INSERT INTO reactions values (?,?,?,?)"""
    return conn.executemany(sql, df.values.tolist())


def read_table(db):
    return pd.read_sql_query("SELECT * from transactions",
                             sqlite3.connect(db))


def stocks_more_than_5():
    sql_statement = """
    SELECT StockCode, Description, Quantity 
    FROM transactions 
    WHERE Quantity > 5
    """
    return sql_statement


def get_invoices():
    sql_statement = """
    SELECT InvoiceNo, COUNT(InvoiceNo) ItemCount, SUM(Quantity) TotalQuantity
    FROM transactions
    GROUP BY InvoiceNo
    ORDER BY ItemCount DESC
    """
    return sql_statement


def white_department(conn):
    sql_statement = """
    SELECT department_id, COUNT() prod_count
    FROM products
    WHERE product_name LIKE 'White %'
    GROUP BY department_id
    ORDER BY prod_count DESC"""
    fin_ser = pd.read_sql_query(sql_statement, conn,
                                index_col='department_id')
    return fin_ser['prod_count']


def count_aisle_products(conn):
    sql_statement = """
    SELECT products.aisle_id AS aisle_id, aisles.aisle AS aisle, COUNT() as product_count
    FROM products
    JOIN aisles
    ON products.aisle_id = aisles.aisle_id
    GROUP BY products.aisle_id
    HAVING product_count < 100
    ORDER BY product_count
    """
    fin_ser = pd.read_sql(sql_statement, con=conn)
    return fin_ser


def concat_cols(conn):
    fin_ans = conn.execute("""
    UPDATE cols SET col3 = col1 || col2 WHERE mod(col2, 2) = 0;
    """)
    return fin_ans


def del_row(conn):
    return conn.execute("DELETE FROM cols WHERE col1 = 'd'")