import re


def get_opcode(intel):
    
    found = re.search('.*, "([_a-zA-Z0-9]+)"',intel)
    if found:
        return found.group(1)
    else:
        return None


def get_dr_opcodes():

    f = open('encoding.h','r')

    dr_opcodes_set = set()
    dr_opcodes_list = []

    for line in f:
        found = re.search('/\*.*OP_([_a-zA-Z0-9]+),.*',line)
        if found:
            dr_opcodes_set.add(found.group(1))
            dr_opcodes_list.append(found.group(1))
    return dr_opcodes_set, dr_opcodes_list

if __name__ == "__main__":

    intel = []
    att = []
    diff_opcodes = set()


    intelf = open('opcode.intel','r')
    attf = open('opcode.att','r')

    intellines = intelf.readlines()
    attlines = attf.readlines()

    for il, al in zip(intellines, attlines):
        if il != al:
            opcode = get_opcode(il)
            if opcode != None:
                diff_opcodes.add(opcode)
    
    intelf.close()
    attf.close()

    dr_opcodes_set, dr_opcodes_list = get_dr_opcodes()
    
    intersect = dr_opcodes_set.intersection(diff_opcodes)

    print diff_opcodes
    print diff_opcodes - dr_opcodes_set
    print len(dr_opcodes_list)
    print len(intersect)

    excluded = ['movsd']
    
    included = 0
    inter_s = 'int change_opcode[] = {'

    for dr_opcode in dr_opcodes_list:
        
        found = re.search('([a-zA-Z0-9]+).*',dr_opcode)
        main_opcode = ''
        if found:
            main_opcode = found.group(1)
        else:
            print dr_opcode

        if dr_opcode in diff_opcodes and dr_opcode not in excluded:
            inter_s += '1'
            included += 1
        elif main_opcode in diff_opcodes and dr_opcode not in excluded:
            inter_s += '1'
            included += 1
        else:
            inter_s += '0'
        if dr_opcode != dr_opcodes_list[-1]:
            inter_s += ','

    inter_s += '};\n'

    print included

    change_opcode_file = open('change_opcode.h','w+')

    change_opcode_file.write(inter_s)
    change_opcode_file.close()


    
