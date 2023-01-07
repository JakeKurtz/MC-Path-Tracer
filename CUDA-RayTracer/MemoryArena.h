#pragma once

#include <list>
#include "Memory.h"

class MemoryArena {
public:

    MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) { };

    ~MemoryArena()
    {
        FreeAligned(currentBlock);
        for (auto& block : usedBlocks)
            FreeAligned(block.second);
        for (auto& block : availableBlocks)
            FreeAligned(block.second);
    };

    void* Alloc(size_t nBytes)
    {
        // Round up nBytes to minimum machine alignment
        nBytes = ((nBytes + 15) & (~15));

        if (currentBlockPos + nBytes > currentAllocSize)
        {
            // Add current block to usedBlockList
            if (currentBlock) {
                usedBlocks.push_back(std::make_pair(currentAllocSize, currentBlock));
                currentBlock = nullptr;
            }
            // Try to get memory block from availableBlocks
            if (!currentBlock) {
                currentAllocSize = std::max(nBytes, blockSize);
                currentBlock = AllocAligned<uint8_t>(currentAllocSize);
            }
            currentBlockPos = 0;
        }
        void* ret = currentBlock + currentBlockPos;
        currentBlockPos += nBytes;
        return ret;
    };

    template<typename T> T* Alloc(size_t n = 1, bool runConstructor = true)
    {
        T* ret = (T*)Alloc(n * sizeof(T));
        if (runConstructor)
            for (size_t i = 0; i < n; ++i)
                new (&ret[i]) T();
        return ret;
    };

    void Reset()
    {
        currentBlockPos = 0;
        availableBlocks.splice(availableBlocks.begin(), usedBlocks);
    };

    size_t TotalAllocated() const
    {
        size_t total = currentAllocSize;
        for (const auto& alloc : usedBlocks)
            total += alloc.first;
        for (const auto& alloc : availableBlocks)
            total += alloc.first;
        return total;
    };

private:
    const size_t blockSize;
    size_t currentBlockPos = 0, currentAllocSize = 0;
    uint8_t* currentBlock = nullptr;
    std::list<std::pair<size_t, uint8_t*>> usedBlocks, availableBlocks;
};