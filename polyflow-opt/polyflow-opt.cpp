#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/raw_ostream.h"

#include "PolyFlow/PolyFlowOps.h"
#include "PolyFlow/PolyFlowDialect.h"
#include "Parsers/SnakemakeParser.hpp"

int main() {

  std::string filename = "example_workflow.snakefile";
  std::ifstream file(filename);
  if (!file) {
      throw std::runtime_error("Cannot open file: " + filename);
  }
  SnakemakeParser parser(file);
  parser.parse();
  parser.print_tree();

  mlir::MLIRContext context;

  context.getOrLoadDialect<polyflow::PolyFlowDialect>();

  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::OpBuilder opBuilder(module.getBodyRegion());

  // Create the ConstantOp with a value of 42.0
  auto dataType = opBuilder.getF64Type();
  auto dataAttribute = mlir::FloatAttr::get(dataType, 42.0);
  polyflow::ConstantOp constantOp = opBuilder.create<polyflow::ConstantOp>(opBuilder.getUnknownLoc(), dataType, dataAttribute);

  module.print(llvm::outs());

  return 0;
}

