import mysql.connector
import struct
import word2vec as w2v
import argparse
import sys
from mysql.connector import errorcode

def create_connection():
    cnx = None
    try:
        cnx = mysql.connector.connect(user='root',password='mysql7788#',database='costmodel',port='43562');
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return cnx


def get_code_data(cnx):
    try:
        cur = cnx.cursor(buffered=True)
        sql = 'SELECT code FROM code'
        data = list()
        cur.execute(sql)
        print cur.rowcount
        row = cur.fetchone()
        i = 0
        while row != None:
            print i
            i = i + 1
            if len(row[0]) % 2 != 0:
                row = cur.fetchone()
                continue
            for i in range(0,len(row[0]),2): 
                slice = row[0][i:i+2]
                convert = struct.unpack('h',slice)
                data.append(int(convert[0]))
            row = cur.fetchone()
    except Exception as e:
        print e
    else:
        return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type',action='store',default=1,type=int)
    args = parser.parse_args(sys.argv[1:])
    
    cnx = create_connection()

    #we have 2 options use the generated tokens as is or build a dataset which will 
    #assign unqiue values to the tokens that are only present in the dataset

    token_data = get_code_data(cnx)
    token_size = 2000 #amount of unique tokens to consider
    cnx.close()

    if args.type == 1:
        #first option
        #build the reverse dictionary
        reverse_dictionary = dict()
        for n, i in enumerate(token_data):
            if i == -1:
                token_data[n] = 0
  
        for token in token_data:
            reverse_dictionary[token] = token
        data = token_data

    elif args.type == 2:

        #second option
        data, count, dictionary, reverse_dictionary = w2v.build_dataset(token_data, token_size)
  
    w2v.train_skipgram(data, len(dictionary), reverse_dictionary)

    
    

