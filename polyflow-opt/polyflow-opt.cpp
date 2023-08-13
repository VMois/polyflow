#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/raw_ostream.h"

#include "Parsers/SnakemakeParser.hpp"

int main() {

  std::string filename = "../example_workflow.snakefile";
  std::ifstream file(filename);
  if (!file) {
      throw std::runtime_error("Cannot open file: " + filename);
  }
  snakemake::Parser parser(file);
  std::unique_ptr<snakemake::ModuleAST> snakemakeModule = parser.parseModule();
  //snakemake::dumpAST(*snakemakeModule);

  mlir::MLIRContext context;
  context.getOrLoadDialect<polyflow::PolyFlowDialect>();

  mlir::ModuleOp m = snakemake::MLIRGen(context).mlirGen(*snakemakeModule);
  m->dump();

  //mlir::OpBuilder builder(&context);
  //mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  // mlir::OpBuilder opBuilder(module.getBodyRegion());

  // std::vector<std::string> strings = {"string1", "string2", "string3"};

  // std::vector<mlir::Attribute> stringAttrs;
  // for (int i = 0; i < 3; i++) {
  //   mlir::Attribute strAttr = mlir::StringAttr::get(&context, strings[i]);
  //   stringAttrs.push_back(strAttr);
  // }

  // mlir::ArrayAttr inputs = mlir::ArrayAttr::get(&context, stringAttrs);

  // polyflow::StepOp stepOp = opBuilder.create<polyflow::StepOp>(
  //   opBuilder.getUnknownLoc(), 
  //   llvm::StringRef("task1"), 
  //   inputs,
  //   inputs,
  //   llvm::StringRef("echo hello world") 
  // );

  return 0;
}

