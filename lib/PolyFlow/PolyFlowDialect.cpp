#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "PolyFlow/PolyFlowDialect.h"
#include "PolyFlow/PolyFlowOps.h"

using namespace mlir;
using namespace polyflow;

//===----------------------------------------------------------------------===//
// Polyflow dialect.
//===----------------------------------------------------------------------===//

#include "PolyFlow/PolyFlowOpsDialect.cpp.inc"

void PolyFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PolyFlow/PolyFlowOps.cpp.inc"
      >();
}
