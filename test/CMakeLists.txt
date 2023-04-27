configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(POLYFLOW_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        polyflow-opt
        )

add_lit_testsuite(check-polyflow "Running the polyflow regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${POLYFLOW_TEST_DEPENDS}
        )
set_target_properties(check-polyflow PROPERTIES FOLDER "Tests")

add_lit_testsuites(POLYFLOW ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${POLYFLOW_TEST_DEPENDS})
