import os
import sys
import subprocess

def execute(command):
	process = subprocess.Popen(args=command, stdout=subprocess.PIPE, shell=True)
	return process.communicate()[0]


def extract_sections(filename):
	section_info = execute("objdump -h " + filename).strip().split('\n')
	section_list = []
	section_vmas = {}
	for line in section_info:
		info = line.strip().split()
		if len(info) < 7:
			continue
		
		section_name = info[1]
		if not(section_name == ".text" or section_name.startswith(".text.")):
			continue	
		try:
			section_size = int(info[2], 16)
			section_offset = int(info[5], 16)
			section_vma = int(info[3], 16)
		except ValueError as verr:
			continue
		section_list += [(section_name, section_size, section_offset)]
		section_vmas[section_name] = section_vma
	return (section_list, section_vmas)

def extract_symbols(filename, section_vma):
	symbols_info = execute("objdump -t " + filename).strip().split('\n')
	symbol_list = []
	for line in symbols_info:
		info = line.strip().split()
		if len(info) < 6:
			continue
		if info[2] != "F":
			continue
		section_name = info[3]
		if not (section_name == ".text" or section_name.startswith(".text.")):
			continue
			
		try:
			symbol_offset_in_section = int(info[0], 16) - section_vma[section_name]
			symbol_size = int(info[4], 16)
		except ValueError as verr:
			continue
		symbol_name = info[5]
		if symbol_name == ".hidden" and len(info) > 6:
			symbol_name = info[6]
		symbol_list += [(symbol_name, section_name, symbol_offset_in_section, symbol_size)]

	return symbol_list			


def create_bin(filename, outfilename, sections_list):
	input_file = open(filename, "rb")
	output_file = open(outfilename, "wb")
	running_offset = 0
	section_offsets = {}
	for section in sections_list:
		try:
			input_file.seek(section[2])
			section_data = input_file.read(section[1])
			if len(section_data) != section[1]:
				raise ValueError("Inadequte length in input file")
			output_file.write(section_data)
			section_offsets[section[0]] = running_offset
			running_offset += section[1]
		except Exception as e:
			print "Error while copying section " + section[0] + " to output file"
			print e
			exit(-1)

	input_file.close()
	output_file.close()
		
	return section_offsets
			

def create_metadata(metadata_filename, symbol_list, section_offsets):
	metadata_file = open(metadata_filename, "w")
	for symbol in symbol_list:
		try:
			actual_offset = section_offsets[symbol[1]] + symbol[2]
		except Exception as e:
			print "Error while creating metadata for function " + symbol[0] 
			print e
			exit(-1)
		metadata_file.write(symbol[0] + "\t" + str(actual_offset) + "\t" + str(symbol[3]) + "\n")
	


def main():
	if len(sys.argv) < 4:
		print "Usage: " + sys.argv[0] + " <input filename> <output binary filename> <output metadata filename>"
		exit(-1)
	input_filename = sys.argv[1]
	output_filename = sys.argv[2]
	output_metadata_filename = sys.argv[3]

	section_list, section_vma = extract_sections(input_filename)
	symbol_list = extract_symbols(input_filename, section_vma)
	section_offsets = create_bin(input_filename, output_filename, section_list)
	create_metadata(output_metadata_filename, symbol_list, section_offsets)
	print "DONE"

if __name__ == "__main__":
	main()
