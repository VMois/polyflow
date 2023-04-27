#ifndef POLYFLOW_OPS
#define POLYFLOW_OPS

include "PolyFlowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : PolyFlow_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = "simple constant operation";

  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);

  let assemblyFormat = "attr-dict `(` $value `)` `:` type($result)";
}

#endif // POLYFLOW_OPS
