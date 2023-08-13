#ifndef POLYFLOW_OPS
#define POLYFLOW_OPS

include "PolyFlowDialect.td"
include "PolyFlowTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def FileLocation : PolyFlow_Op<"FileLocation", [Pure]> {
  let summary = "An input value to a file.";
  let description = "A single step in the workflow execution.";

  let arguments = (ins StrAttr:$value);

  let results = (outs PolyFlow_StringType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def StepOp : PolyFlow_Op<"Step", []> {
  let summary = "A single step in the workflow execution.";
  let description = "A single step in the workflow execution.";

  let arguments = (ins 
    StrAttr:$name,
    Variadic<PolyFlow_StringType>:$inputs
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$name `(` $inputs `)` attr-dict `(` type($inputs) `)` `:` $body";
}

#endif // POLYFLOW_OPS
