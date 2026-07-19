/* mmap_file.cpp  --  read-only memory-mapping of a binary file (Windows + POSIX).
 *
 * Foundation for OUT-OF-CORE entry streaming: the streaming-B2 SpMV reads the
 * per-entry table from a borrowed raw host pointer (s_h_src_idx etc.) and
 * cudaMemcpy's tile slices H2D. If that pointer points at a memory-mapped file
 * instead of a resident MATLAB array, the OS pages the touched tiles from NVMe
 * on demand (reclaimable page cache) -> the resident host RAM for the entry
 * table drops to ~0, with no change to the SpMV kernel itself.
 *
 *   [ptr, nbytes] = mmap_file('open',  filename)   % map read-only, return base ptr (uint64)
 *   data          = mmap_file('read',  ptr, byteoffset, count, classname) % copy a slice (verify/test)
 *                   mmap_file('close', ptr)         % unmap + close handles
 *
 * classname in {'int32','uint32','uint16','uint8'}. The returned ptr is a raw
 * process address (uint64) to be handed to init_skel_stream as the entry source.
 *
 * PORTABILITY (Linux port, June 2026): the Windows branch uses CreateFileA /
 * CreateFileMappingA / MapViewOfFile; the POSIX branch (#else) uses open /
 * fstat / mmap(PROT_READ, MAP_PRIVATE) / munmap / close. The MEX contract and
 * every consumer (spill_entries_mmap, entry_skeleton_ondisk, host_ptr_arg in
 * the CUDA kernel) are byte-identical across platforms.
 *
 * Build:  mex mmap_file.cpp        (Windows -> .mexw64, Linux -> .mexa64)
 *
 * ================================================================
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 * Licensed under the Apache License, Version 2.0.
 * ================================================================
 */
#include "mex.h"
#include <map>
#include <utility>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
/* base ptr -> (file handle, mapping handle), so 'close' only needs the ptr. */
static std::map<void *, std::pair<HANDLE, HANDLE> > g_maps;
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
/* base ptr -> (fd, mapped size), so 'close' only needs the ptr. */
struct MmapEntry { int fd; size_t size; };
static std::map<void *, MmapEntry> g_maps;
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("mmap_file:args",
            "First arg must be a command: 'open' | 'read' | 'close'.");
    char cmd[16] = {0};
    mxGetString(prhs[0], cmd, sizeof(cmd));

    if (strcmp(cmd, "open") == 0) {
        if (nrhs < 2 || !mxIsChar(prhs[1]))
            mexErrMsgIdAndTxt("mmap_file:open",
                "Usage: [ptr,nbytes] = mmap_file('open', filename)");
        char *fn = mxArrayToString(prhs[1]);
#ifdef _WIN32
        HANDLE hFile = CreateFileA(fn, GENERIC_READ, FILE_SHARE_READ, NULL,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        mxFree(fn);
        if (hFile == INVALID_HANDLE_VALUE)
            mexErrMsgIdAndTxt("mmap_file:open", "CreateFileA failed (file missing?).");
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) { CloseHandle(hFile);
            mexErrMsgIdAndTxt("mmap_file:open", "GetFileSizeEx failed."); }
        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { CloseHandle(hFile);
            mexErrMsgIdAndTxt("mmap_file:open", "CreateFileMapping failed."); }
        void *p = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!p) { CloseHandle(hMap); CloseHandle(hFile);
            mexErrMsgIdAndTxt("mmap_file:open", "MapViewOfFile failed."); }
        g_maps[p] = std::make_pair(hFile, hMap);
        uint64_T nbytes = (uint64_T)sz.QuadPart;
#else
        int fd = open(fn, O_RDONLY);
        mxFree(fn);
        if (fd < 0)
            mexErrMsgIdAndTxt("mmap_file:open", "open failed (file missing?).");
        struct stat sb;
        if (fstat(fd, &sb) < 0) { close(fd);
            mexErrMsgIdAndTxt("mmap_file:open", "fstat failed."); }
        size_t fsz = (size_t)sb.st_size;
        void *p = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) { close(fd);
            mexErrMsgIdAndTxt("mmap_file:open", "mmap failed."); }
        MmapEntry me; me.fd = fd; me.size = fsz;
        g_maps[p] = me;
        uint64_T nbytes = (uint64_T)fsz;
#endif
        plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
        *(uint64_T *)mxGetData(plhs[0]) = (uint64_T)(uintptr_t)p;
        if (nlhs > 1) {
            plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
            *(uint64_T *)mxGetData(plhs[1]) = nbytes;
        }

    } else if (strcmp(cmd, "read") == 0) {
        if (nrhs < 5)
            mexErrMsgIdAndTxt("mmap_file:read",
                "Usage: data = mmap_file('read', ptr, byteoffset, count, classname)");
        uint64_T pv  = *(uint64_T *)mxGetData(prhs[1]);
        uint64_T off = (uint64_T)mxGetScalar(prhs[2]);
        uint64_T cnt = (uint64_T)mxGetScalar(prhs[3]);
        char cls[16] = {0};
        mxGetString(prhs[4], cls, sizeof(cls));
        char *base = (char *)(uintptr_t)pv + off;
        if (strcmp(cls, "int32") == 0) {
            plhs[0] = mxCreateNumericMatrix((mwSize)cnt, 1, mxINT32_CLASS, mxREAL);
            memcpy(mxGetData(plhs[0]), base, (size_t)cnt * 4);
        } else if (strcmp(cls, "uint32") == 0) {
            plhs[0] = mxCreateNumericMatrix((mwSize)cnt, 1, mxUINT32_CLASS, mxREAL);
            memcpy(mxGetData(plhs[0]), base, (size_t)cnt * 4);
        } else if (strcmp(cls, "uint16") == 0) {
            plhs[0] = mxCreateNumericMatrix((mwSize)cnt, 1, mxUINT16_CLASS, mxREAL);
            memcpy(mxGetData(plhs[0]), base, (size_t)cnt * 2);
        } else if (strcmp(cls, "uint8") == 0) {
            plhs[0] = mxCreateNumericMatrix((mwSize)cnt, 1, mxUINT8_CLASS, mxREAL);
            memcpy(mxGetData(plhs[0]), base, (size_t)cnt * 1);
        } else {
            mexErrMsgIdAndTxt("mmap_file:read", "Unsupported class '%s'.", cls);
        }

    } else if (strcmp(cmd, "close") == 0) {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("mmap_file:close", "Usage: mmap_file('close', ptr)");
        uint64_T pv = *(uint64_T *)mxGetData(prhs[1]);
        void *p = (void *)(uintptr_t)pv;
#ifdef _WIN32
        std::map<void *, std::pair<HANDLE, HANDLE> >::iterator it = g_maps.find(p);
        if (it != g_maps.end()) {
            UnmapViewOfFile(p);
            CloseHandle(it->second.second);
            CloseHandle(it->second.first);
            g_maps.erase(it);
        }
#else
        std::map<void *, MmapEntry>::iterator it = g_maps.find(p);
        if (it != g_maps.end()) {
            munmap(p, it->second.size);
            close(it->second.fd);
            g_maps.erase(it);
        }
#endif
    } else {
        mexErrMsgIdAndTxt("mmap_file:cmd", "Unknown command '%s'.", cmd);
    }
}
