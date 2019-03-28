#include "disassemble.h"
#include "assert.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  Disassembler disassembler(Disassembler::Intel);
  std::vector<uint8_t> x = {
    0xff, 0x25, 0x7a, 0x34, 0x20, 0x00,
    0x68, 0x19, 0x00, 0x00, 0x00
  };
  std::string hex =
    "554889e54157415641554154534889fb4883ec680f3148c1e22089c04809c24989d54885c0";
  std::cerr << "DISASSEMBLING FROM HEX STR\n";
  auto y = hexToBinary(hex);
  std::string out;
  disassembler.disassemble(y, out);
  std::cerr << out << '\n';

  std::cerr << "DISASSEMBLING FROM BYTE ARRAY\n";
  disassembler.disassemble(x, out);
  std::cerr << out << '\n';
}
