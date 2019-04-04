#include "disassembler.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.inc"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"

using namespace llvm;

static uint8_t parseHexDigit(char c) {
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 0xa;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 0xa;
  return c - '0';
}

std::vector<uint8_t> hexToBinary(StringRef hex) {
  assert(hex.size() % 2 == 0);
  std::vector<uint8_t> binary(hex.size() / 2);
  for (int i = 0, e = binary.size(); i != e; i++) {
    uint8_t hi = parseHexDigit(hex[i*2]);
    uint8_t lo = parseHexDigit(hex[i*2 + 1]);
    binary[i] = hi * 16 + lo;
  }
  return binary;
}


Disassembler::Disassembler(AsmSyntax syntax) {
  tripleName = Triple::normalize(sys::getDefaultTargetTriple());

  // Initialize targets and assembly printers/parsers.
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllDisassemblers();

  std::string errorMsg;
  triple = Triple(tripleName);
  target = TargetRegistry::lookupTarget(""/*arch name*/, triple,
      errorMsg);
  if (!target) {
    errs() << "Unable to find target: " << errorMsg << '\n';
    abort();
  }

  MRI = target->createMCRegInfo(tripleName);
  assert(MRI && "Unable to create target register info!");

  MAI = target->createMCAsmInfo(*MRI, tripleName);
  assert(MAI && "Unable to create target asm info!");

  // Set up the MCContext for creating symbols and MCExpr's.
  ctx = new MCContext(MAI, MRI, nullptr);

  std::string FeaturesStr;
  std::string MCPU;
  STI = target->createMCSubtargetInfo(tripleName, MCPU, FeaturesStr);

  disassembler = target->createMCDisassembler(*STI, *ctx);

  MCII = target->createMCInstrInfo();

  unsigned asmVariant = (syntax != ATT);
  IP = target->createMCInstPrinter(triple, asmVariant,
      *MAI, *MCII, *MRI);
  IP->setPrintImmHex(true);
}

bool Disassembler::disassemble(const std::vector<uint8_t> &binary,
    std::string &output) const {
  uint64_t size;
  uint64_t index;

  ArrayRef<uint8_t> data = binary;

  output.clear();
  raw_string_ostream os(output);

  for (index = 0; index < data.size(); index += size) {
    MCInst Inst;

    auto status = disassembler->getInstruction(Inst, size, data.slice(index), index,
                              /*REMOVE*/ nulls(), nulls());
    if (status != MCDisassembler::Success)
      return false;

    IP->printInst(&Inst, os, ""/*annotation*/, *STI);
    os << '\n';
  }

  return true;
}
