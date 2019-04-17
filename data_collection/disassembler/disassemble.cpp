#include "disassembler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/CommandLine.h"

#include "assert.h"


using namespace llvm;

static cl::opt<std::string> inputFileName(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<Disassembler::AsmSyntax> syntaxType(cl::desc("Choose output syntax type:"),
                                                   cl::values(clEnumValN(Disassembler::Intel, "intel", "Output Intel syntax (default)"),
                                                              clEnumValN(Disassembler::ATT, "att" , "Output AT&T ayntax")));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  ErrorOr<std::unique_ptr<MemoryBuffer>> bufferPtr =
      MemoryBuffer::getFileOrSTDIN(inputFileName);

  if (std::error_code EC = bufferPtr.getError()) {
    errs() << inputFileName << ": " << EC.message() << '\n';
    return 1;
  }

  StringRef buffer = bufferPtr->get()->getBuffer();

  Disassembler disassembler(syntaxType);

  StringRef hex = buffer.trim();
  auto y = hexToBinary(hex);
  std::string out;
  bool ok = disassembler.disassemble(y, out);
  if (!ok) {
    errs() << "Disassembly failed\n";
    return 1;
  }
  outs() << out << '\n';
  return 0;
}
