/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/memory/memory_stats.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace llm_on_edge::memory
{

void MemoryStats::bump(MemoryType t, DiffType delta) noexcept
{
    switch (t)
    {
    case MemoryType::kGPU:
        m_gpu.fetch_add(static_cast<SizeType>(delta), std::memory_order_relaxed);
        m_last_diff_gpu.store(delta, std::memory_order_relaxed);
        break;
    case MemoryType::kCPU:
        m_cpu.fetch_add(static_cast<SizeType>(delta), std::memory_order_relaxed);
        m_last_diff_cpu.store(delta, std::memory_order_relaxed);
        break;
    case MemoryType::kPINNED:
        m_pinned.fetch_add(static_cast<SizeType>(delta), std::memory_order_relaxed);
        m_last_diff_pinned.store(delta, std::memory_order_relaxed);
        break;
    }
}

void MemoryStats::on_allocate(MemoryType t, SizeType bytes) noexcept
{
    bump(t, static_cast<DiffType>(bytes));
}

void MemoryStats::on_deallocate(MemoryType t, SizeType bytes) noexcept
{
    bump(t, -static_cast<DiffType>(bytes));
}

std::string MemoryStats::bytes_to_string(SizeType bytes, int precision)
{
    static char const* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4)
    {
        v /= 1024.0;
        ++u;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v << " " << units[u];
    return oss.str();
}

std::string MemoryStats::to_string() const
{
    std::ostringstream oss;
    oss << "MemoryStats{gpu=" << bytes_to_string(current_gpu()) << " cpu=" << bytes_to_string(current_cpu())
        << " pinned=" << bytes_to_string(current_pinned()) << "}";
    return oss.str();
}

} // namespace llm_on_edge::memory
