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

int main() {
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

