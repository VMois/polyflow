#ifndef STANDALONE_TYPES
#define STANDALONE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "PolyFlowDialect.td"


class PolyFlow_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PolyFlow_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


def PolyFlow_StringType : PolyFlow_Type<"Polyflow_Str", "polyflow_str"> {
  let summary = "A string";
  let description = "Custom type in standalone dialect";
  //let parameters = (ins StringRefParameter<"the custom value">:$value);
  //let assemblyFormat = "`<` $value `>`";
}

#endif
