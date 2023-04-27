#ifndef POLYFLOW_POLYFLOWOPS_H
#define POLYFLOW_POLYFLOWOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "PolyFlow/PolyFlowOps.h.inc"

#endif // POLYFLOW_POLYFLOWOPS_H
