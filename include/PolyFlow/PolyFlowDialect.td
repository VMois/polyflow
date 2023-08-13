#ifndef POLYFLOW_DIALECT
#define POLYFLOW_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Polyflow dialect definition.
//===----------------------------------------------------------------------===//

def PolyFlow_Dialect : Dialect {
    let name = "polyflow";
    let summary = "An out-of-tree MLIR dialect for workflow languages";
    let description = [{
        This dialect is an experimental project to unify workflow languages.
    }];
    let cppNamespace = "::polyflow";
    let hasConstantMaterializer = 0;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base polyflow operation definition.
//===----------------------------------------------------------------------===//

class PolyFlow_Op<string mnemonic, list<Trait> traits = []> :
        Op<PolyFlow_Dialect, mnemonic, traits>;

#endif // POLYFLOW_DIALECT
