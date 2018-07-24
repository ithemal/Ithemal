from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import common.utilities as ut
from tqdm import tqdm
import subprocess
import os
import re
import time
import argparse



if __name__ == "__main__":


    costmodel_cnx = ut.create_connection('costmodel')
    static_cnx = ut.create_connection('static')
    teststatic_cnx = ut.create_connection('teststatic')

    #config
    sql = 'select * from config'
    print sql
    config_data = ut.execute_query(teststatic_cnx, sql, True)

    sql = 'insert into config values (%s, %s, %s)'
    print sql
    try:
        ut.execute_many(costmodel_cnx, sql, config_data)
    except Exception as e:
        print e
    else:
        costmodel_cnx.commit()
    
    #static
    rows_sql = 'program, config_id, rel_addr, code_intel, code_att, code_token'
    sql = 'SELECT ' + rows_sql + ' from code'
    print sql
    static_data = ut.execute_query(teststatic_cnx, sql, True)


    # for row in tqdm(static_data):
    #     sql = 'insert into code (' + rows_sql + ') values (\''
    #     sql += row[0] + '\', '
    #     sql += str(row[1]) + ', '
    #     sql += str(row[2]) + ', '
    #     sql += '\'' + row[3] + '\', '
    #     sql += '\'' + row[4] + '\', '
    #     sql += '\'' + row[5] + '\')'
    #     try:
    #         ut.execute_query(costmodel_cnx, sql, False)
    #     except Exception as e:
    #         pass

    # costmodel_cnx.commit()


    #times
    for row in tqdm(static_data):
        
        try:
            sql = 'select time from code where program=\'' + row[0] + '\' and config_id=' + str(row[1]) + ' and rel_addr=' + str(row[2])
            times = ut.execute_query(static_cnx, sql, True)
            if times != None:
                assert len(times) == 1
            
                #get code id
                if times[0][0] != None:
                    sql = 'select code_id from code where program=\'' + row[0] + '\' and config_id=' + str(row[1]) + ' and rel_addr=' + str(row[2])
                    code_id = ut.execute_query(costmodel_cnx, sql, True)
                    if code_id != None:
                        assert len(code_id) == 1

                        if code_id[0][0] != None:
                            sql = 'insert into times (code_id, arch, kind, time) values ('
                            sql += str(code_id[0][0]) + ', '
                            sql += '1, \'actual\', '
                            sql += str(times[0][0]) + ')'

                            ut.execute_query(costmodel_cnx, sql, False)
                    
        except Exception as e:
            pass

    costmodel_cnx.commit()

    
    static_cnx.close()
    teststatic_cnx.close()
    costmodel_cnx.close()
