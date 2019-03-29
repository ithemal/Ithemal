#ifndef DISASSEMBLE_H
#define DISASSEMBLE_H

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
#include "llvm/MC/MCDisassembler/MCDisassembler.h"

#include <vector>
#include <string>
#include <memory>

std::vector<uint8_t> hexToBinary(llvm::StringRef hex);

class Disassembler {
  const llvm::Target *target;
  llvm::Triple triple;
  std::string tripleName;
  llvm::MCRegisterInfo *MRI;
  llvm::MCAsmInfo *MAI;
  llvm::MCContext *ctx;
  llvm::MCSubtargetInfo *STI;
  llvm::MCDisassembler *disassembler;
  llvm::MCInstrInfo *MCII;
  llvm::MCInstPrinter *IP;

public:
  enum AsmSyntax { Intel, ATT };

  Disassembler(AsmSyntax syntax);
  bool disassemble(
    const std::vector<uint8_t> &binary,
    std::string &output) const;
};

#endif
