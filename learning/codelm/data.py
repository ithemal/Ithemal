
import mysql.connector

def create_connection():
    try:
        cnx = mysql.connector.connect(user='root',password='mysql7788#',database='costmodel',port='43562');
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()


def get_code_samples():
