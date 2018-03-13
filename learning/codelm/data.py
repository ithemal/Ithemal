import mysql.connector
import struct
import word2vec as w2v
import argparse
import sys
from mysql.connector import errorcode
import re
import os
from tempfile import gettempdir
import matplotlib

matplotlib.use('Agg')


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


def get_opcode_opnd_dict(opcode_start, opnd_start):
    sym_dict = dict()
    with open('encoding.h','r') as f:
        opcode_num = opcode_start
        opnd_num = opnd_start
        for line in f:
            opcode_re = re.search('/\*.*\*/.*OP_([a-zA-Z_0-9]+),.*', line)
            if opcode_re != None:
                sym_dict[opcode_num] = opcode_re.group(1)
                opcode_num = opcode_num + 1
            opnd_re = re.search('.*DR_([A-Za-z_0-9]+),.*', line)
            if opnd_re != None:
                sym_dict[opnd_num] = opnd_re.group(1)
                opnd_num = opnd_num + 1
        f.close()

    return sym_dict


def get_code_data(cnx, format):
    try:
        cur = cnx.cursor(buffered=True)
        sql = 'SELECT code FROM code'
        data = list()
        cur.execute(sql)
        print cur.rowcount
        row = cur.fetchone()
        while row != None:
            if format == 'text':
                for value in row[0].split(','):
                    if value != '':
                        data.append(int(value))
            elif format == 'bin':
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

def read_offsets():
    offsets_filename = '/data/scratch/charithm/projects/cmodel/database/offsets.txt'
    offsets = list()
    with open(offsets_filename,'r') as f:
        for line in f:
            for value in line.split(','):
                offsets.append(int(value))
        f.close()
    assert len(offsets) == 5
    return offsets
            

if __name__ == "__main__":

    offsets = read_offsets()
    sym_dict = get_opcode_opnd_dict(opcode_start = offsets[0],opnd_start = offsets[1])
   
    sym_dict[offsets[2]] = 'int_immed'
    sym_dict[offsets[3]] = 'float_immed'

    parser = argparse.ArgumentParser()
    parser.add_argument('--type',action='store',default=2,type=int)
    parser.add_argument('--format',action='store',default='text',type=str);
    args = parser.parse_args(sys.argv[1:])
    
    cnx = create_connection()

    #we have 2 options use the generated tokens as is or build a dataset which will 
    #assign unqiue values to the tokens that are only present in the dataset

    token_data = get_code_data(cnx,args.format)
    token_size = 2000 #amount of unique tokens to consider

    print len(token_data)

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
  
    print len(dictionary)

    final_embeddings = w2v.train_skipgram(data, len(dictionary), reverse_dictionary, sym_dict, offsets[4])

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 200
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels_token = [reverse_dictionary[i] for i in xrange(plot_only)]
        labels = list()
        for token in labels_token:
            if token in sym_dict.keys():
                labels.append(sym_dict[token])
            else:
                labels.append('mem_' + str(token - offsets[4]))
        w2v.plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
        
    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)

