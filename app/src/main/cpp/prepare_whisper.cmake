# Prepares a patched whisper.cpp source tree for the BPDQ2 2-bit model.
#
#   1. downloads whisper.cpp at the exact commit the BPDQ2 patch was made against
#   2. downloads the author's patch from Hugging Face and checks its SHA-256
#   3. applies it (git apply, or patch -p1 as a fallback)
#   4. optionally swaps the patch's scalar CPU dot product for the NEON one in
#      bpdq2_fast.inc (falls back to the scalar kernel if the anchors are missing)
#
# Inputs (set before include()):
#   BPDQ_DEPS_DIR       where to keep the prepared tree (shared between ABIs/variants)
#   BPDQ_FAST_INC       absolute path of bpdq2_fast.inc
#   BPDQ_FAST_KERNEL    ON/OFF
# Output:
#   WHISPER_SOURCE_DIR  path of the prepared, patched whisper.cpp tree

set(WHISPER_COMMIT     "5670d5c0bbcb148feabef84400a07cfca9aa3b30")
set(WHISPER_TARBALL    "https://github.com/ggml-org/whisper.cpp/archive/${WHISPER_COMMIT}.tar.gz")
set(BPDQ_PATCH_URL     "https://huggingface.co/diegogranziol/whisper-large-v3-turbo-bpdq-2bit-qat/resolve/main/runtime-source/whisper-bpdq2.patch")
set(BPDQ_PATCH_SHA256  "0967c05ed145ebb417e339d4d9fc56d4c7be6e2db928e3b67114d103e756b688")

if(BPDQ_FAST_KERNEL)
    file(SHA256 "${BPDQ_FAST_INC}" _fast_sha)
    string(SUBSTRING "${_fast_sha}" 0 12 _fast_tag)
else()
    set(_fast_tag "scalar")
endif()
string(SUBSTRING "${WHISPER_COMMIT}" 0 10 _commit_short)
string(SUBSTRING "${BPDQ_PATCH_SHA256}" 0 10 _patch_short)

set(WHISPER_SOURCE_DIR "${BPDQ_DEPS_DIR}/whisper.cpp-${_commit_short}-${_patch_short}-${_fast_tag}")
set(_stamp "${WHISPER_SOURCE_DIR}/.bpdq-prepared")

file(MAKE_DIRECTORY "${BPDQ_DEPS_DIR}")
# Gradle may configure several variants at once; only one of them prepares the tree.
file(LOCK "${BPDQ_DEPS_DIR}/.lock" GUARD FILE TIMEOUT 1800)

if(NOT EXISTS "${_stamp}")
    message(STATUS "BPDQ2: preparing patched whisper.cpp in ${WHISPER_SOURCE_DIR}")
    file(REMOVE_RECURSE "${WHISPER_SOURCE_DIR}" "${BPDQ_DEPS_DIR}/_extract")

    # --- 1. whisper.cpp source -------------------------------------------------
    set(_tarball "${BPDQ_DEPS_DIR}/whisper.cpp-${WHISPER_COMMIT}.tar.gz")
    if(NOT EXISTS "${_tarball}")
        message(STATUS "BPDQ2: downloading whisper.cpp ${WHISPER_COMMIT}")
        file(DOWNLOAD "${WHISPER_TARBALL}" "${_tarball}.tmp" STATUS _st TLS_VERIFY ON)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            file(REMOVE "${_tarball}.tmp")
            message(FATAL_ERROR "BPDQ2: could not download ${WHISPER_TARBALL}: ${_st}")
        endif()
        file(RENAME "${_tarball}.tmp" "${_tarball}")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${_tarball}" DESTINATION "${BPDQ_DEPS_DIR}/_extract")
    file(RENAME "${BPDQ_DEPS_DIR}/_extract/whisper.cpp-${WHISPER_COMMIT}" "${WHISPER_SOURCE_DIR}")
    file(REMOVE_RECURSE "${BPDQ_DEPS_DIR}/_extract")

    # --- 2. the BPDQ2 patch ----------------------------------------------------
    set(_patch "${BPDQ_DEPS_DIR}/whisper-bpdq2-${_patch_short}.patch")
    if(NOT EXISTS "${_patch}")
        message(STATUS "BPDQ2: downloading runtime patch from Hugging Face")
        file(DOWNLOAD "${BPDQ_PATCH_URL}" "${_patch}.tmp"
             EXPECTED_HASH SHA256=${BPDQ_PATCH_SHA256} STATUS _st TLS_VERIFY ON)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            file(REMOVE "${_patch}.tmp")
            message(FATAL_ERROR "BPDQ2: could not download/verify ${BPDQ_PATCH_URL}: ${_st}")
        endif()
        file(RENAME "${_patch}.tmp" "${_patch}")
    endif()

    # --- 3. apply it -----------------------------------------------------------
    find_program(_git git)
    find_program(_patchexe patch)
    if(_git)
        execute_process(COMMAND "${_git}" apply --whitespace=nowarn -p1 "${_patch}"
                        WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
                        RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    elseif(_patchexe)
        execute_process(COMMAND "${_patchexe}" -p1 --forward --batch -i "${_patch}"
                        WORKING_DIRECTORY "${WHISPER_SOURCE_DIR}"
                        RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    else()
        message(FATAL_ERROR "BPDQ2: need 'git' or 'patch' on PATH to apply the runtime patch")
    endif()
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "BPDQ2: applying the patch failed (${_rc}):\n${_err}")
    endif()

    # --- 4. faster NEON dot product ---------------------------------------------
    if(BPDQ_FAST_KERNEL)
        set(_cpu_c "${WHISPER_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.c")
        file(READ "${_cpu_c}" _src)

        # Anchor 1: the patch's scalar kernel definition (we include ours just before it).
        set(_def_re   "static void ggml_vec_dot_bpdq2_f32\\(int n,")
        # Anchor 2: the BPDQ2 entry of the CPU type-traits table (no nested braces inside).
        set(_entry_re "\\[GGML_TYPE_BPDQ2\\][ \t]*=[ \t]*{[^}]*}")
        set(_vdot_re  "\\.vec_dot([ \t]*)=([ \t]*)ggml_vec_dot_bpdq2_f32([ \t]*),")
        set(_nrows_re "\\.nrows([ \t]*)=([ \t]*)1([ \t]*),")

        string(REGEX MATCHALL "${_def_re}"   _defs    "${_src}")
        string(REGEX MATCHALL "${_entry_re}" _entries "${_src}")
        list(LENGTH _defs    _n_defs)
        list(LENGTH _entries _n_entries)

        set(_ok FALSE)
        if(_n_defs EQUAL 1 AND _n_entries EQUAL 1)
            set(_entry "${_entries}")
            string(REGEX MATCHALL "${_vdot_re}"  _vdots  "${_entry}")
            string(REGEX MATCHALL "${_nrows_re}" _nrowss "${_entry}")
            list(LENGTH _vdots  _n_vdots)
            list(LENGTH _nrowss _n_nrows)
            if(_n_vdots EQUAL 1 AND _n_nrows EQUAL 1)
                # Register the NEON kernel, and let mul_mat hand it 2x2 tiles.
                string(REGEX REPLACE "${_vdot_re}"  ".vec_dot\\1=\\2ggml_vec_dot_bpdq2_f32_fast\\3," _new_entry "${_entry}")
                string(REGEX REPLACE "${_nrows_re}" ".nrows\\1=\\22\\3,"                             _new_entry "${_new_entry}")
                string(REPLACE "${_entry}" "${_new_entry}" _src "${_src}")

                string(REGEX REPLACE "${_def_re}"
                    "#include \"bpdq2_fast.inc\"\n\n// unused once the fast kernel is registered\nGGML_ATTRIBUTE_UNUSED_BPDQ2 static void ggml_vec_dot_bpdq2_f32(int n,"
                    _src "${_src}")
                # keep -Wunused-function quiet for the now-unregistered scalar kernel
                string(PREPEND _src "#if defined(__GNUC__)\n#define GGML_ATTRIBUTE_UNUSED_BPDQ2 __attribute__((unused))\n#else\n#define GGML_ATTRIBUTE_UNUSED_BPDQ2\n#endif\n")

                file(COPY "${BPDQ_FAST_INC}" DESTINATION "${WHISPER_SOURCE_DIR}/ggml/src/ggml-cpu")
                file(WRITE "${_cpu_c}" "${_src}")
                set(_ok TRUE)
                message(STATUS "BPDQ2: NEON dot product installed (2x2 tiles)")
            endif()
        endif()
        if(NOT _ok)
            message(WARNING "BPDQ2: fast-kernel anchors not found (defs=${_n_defs}, entries=${_n_entries}); "
                            "using the patch's scalar kernel, which is correct but slower")
        endif()
    endif()

    file(WRITE "${_stamp}" "commit=${WHISPER_COMMIT}\npatch=${BPDQ_PATCH_SHA256}\nfast=${_fast_tag}\n")
    message(STATUS "BPDQ2: whisper.cpp ready")
endif()

file(LOCK "${BPDQ_DEPS_DIR}/.lock" RELEASE)
