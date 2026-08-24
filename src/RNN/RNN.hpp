#pragma once
#ifndef RNN_HPP
#define RNN_HPP

#include "include/sac.hpp"
#include "include/init.hpp"
#include "include/run.hpp"

using namespace _mi;

// 常用头文件

// ---------- 1. 默认标配 ----------
#include <iostream>

// ---------- 2. IOs 增强（需 #define IOs） ----------
#ifdef IOs
    #include <iomanip>
    #include <fstream>
    #include <sstream>
#endif

// ---------- 3. List 容器（需 #define List） ----------
#ifdef List
    #include <vector>
    #include <array>
#endif

// ---------- 4. 系统平台 API（需 #define _sys） ----------
#ifdef _sys
    #ifdef _WIN32
        #define WIN32_LEAN_AND_MEAN
        #define NOMINMAX
        #include <windows.h>
        #include <io.h>
        #include <fcntl.h>

        #ifdef _u8
            SetConsoleOutputCP(CP_UTF8);
            SetConsoleCP(CP_UTF8);
            _setmode(_fileno(stdout), _O_U8TEXT);
            _setmode(_fileno(stderr), _O_U8TEXT);
        #endif
    #elif __unix__ || __APPLE__
        #include <unistd.h>
    #else
        #error "未知平台"
    #endif
#endif

#endif
