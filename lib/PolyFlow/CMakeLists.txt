add_mlir_dialect_library(MLIRPolyFlow
		PolyFlowDialect.cpp
		PolyFlowOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/PolyFlow

        DEPENDS
        MLIRPolyFlowOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
