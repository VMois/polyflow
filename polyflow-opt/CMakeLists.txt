get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRPolyFlow
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIROptLib
        SnakemakeParser
        )
add_llvm_executable(polyflow-opt polyflow-opt.cpp)

llvm_update_compile_flags(polyflow-opt)
target_link_libraries(polyflow-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(polyflow-opt)
