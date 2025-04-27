import sqlite3

conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Выбираем все данные из таблицы 'users'
cursor.execute("SELECT * FROM transactions;")
rows = cursor.fetchall()

print("\nДанные в таблице transactions:")
for row in rows:
    print(row)


cursor.execute(f'''WITH ordered_transactions AS (
    SELECT
        t.customer_id,
        t.id AS transaction_id,
        t.amount_rur,
        t.transaction_dttm,
        LAG(t.amount_rur) OVER (PARTITION BY t.customer_id ORDER BY t.transaction_dttm) AS prev_amount,
        LEAD(t.amount_rur) OVER (PARTITION BY t.customer_id ORDER BY t.transaction_dttm) AS next_amount
    FROM
        transactions t
    WHERE
        t.success_flg = 1
        AND t.amount_rur <= 1000000
)

SELECT
    c.name AS customer_name,
    ot.transaction_id
FROM
    ordered_transactions ot
JOIN
    customer c
    ON ot.customer_id = c.id
WHERE
    ot.prev_amount IS NOT NULL
    AND ot.next_amount IS NOT NULL
    AND ot.amount_rur > ot.prev_amount
    AND ot.amount_rur > ot.next_amount
ORDER BY
    c.name,
    ot.transaction_id;
''')
rows = cursor.fetchall()

for row in rows:
    print(row)  # Выводим каждую строку

conn.close()