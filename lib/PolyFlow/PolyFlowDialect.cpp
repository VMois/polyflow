#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PolyFlow/PolyFlowDialect.h"
#include "PolyFlow/PolyFlowTypes.h"
#include "PolyFlow/PolyFlowOps.h"

using namespace mlir;
using namespace polyflow;

//===----------------------------------------------------------------------===//
// Polyflow dialect.
//===----------------------------------------------------------------------===//

#include "PolyFlow/PolyFlowOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PolyFlow/PolyFlowOpsTypes.cpp.inc"


void PolyFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PolyFlow/PolyFlowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "PolyFlow/PolyFlowOpsTypes.cpp.inc"
      >();
}
