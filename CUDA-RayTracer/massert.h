/***************************************************************************************
    *    Title: Custom Messages in Assert
    *    Author: Eugene Magdalits
    *    Availability: https://stackoverflow.com/questions/3692954/add-custom-messages-in-assert
    *
***************************************************************************************/

#pragma once
#include <iostream>

#ifndef NDEBUG
#   define m_assert(Expr, Msg) \
    __m_assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define m_assert(Expr, Msg) ;
#endif

extern void __m_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);