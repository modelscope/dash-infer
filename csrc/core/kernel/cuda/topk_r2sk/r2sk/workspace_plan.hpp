#pragma once
#include "batch_core.hpp"
#include <array>
#include <limits>

namespace r2sk {
namespace detail {
template<class Value> struct KernelJobT {
    ValueView<Value> src{},pivot{};
    ValueBuffer<Value> G{},E{};
    uint32_t n=0,rank=0,boundary=0,gcap=0,ecap=0,tile=0,width=0;
    int pass=0;
    uint32_t* hist=nullptr;
    uint32_t* counters=nullptr;
    uint32_t* scale_unsafe=nullptr;
    Decision* decision=nullptr;
};
inline uint32_t batch_descriptor_slots(uint32_t k,uint32_t limit){
    uint32_t levels=0;for(uint64_t width=4096;width<std::max(k,limit);width*=2)++levels;
    return padded(6*(12+3*levels)+1);
}
struct ArenaRegion {uint64_t offset=0,bytes=0;};
struct ArenaPlanner {
    static constexpr uint64_t alignment=256;
    uint64_t bytes=0;
    ArenaRegion add(uint64_t count,uint64_t element){
        const auto max=std::numeric_limits<uint64_t>::max();
        if(bytes>max-(alignment-1) || (element && count>max/element))
            throw std::overflow_error("workspace size overflow");
        uint64_t start=(bytes+alignment-1)&~(alignment-1),size=count*element;
        if(size>max-start)throw std::overflow_error("workspace size overflow");
        bytes=start+size;return {start,size};
    }
    uint64_t finish(){add(0,1);return bytes;}
};
struct BatchArenaLayout {
    // c[0], c[1], winners, potential, sort scratch, winner sort scratch.
    std::array<ArenaRegion,6> values{},indices{};
    ArenaRegion histogram,metadata,counters,jobs;
    uint64_t bytes=0;
    uint32_t descriptor_slots=0;
};
template<class Value> BatchArenaLayout batch_arena_layout(uint32_t batch,uint32_t n,
    uint32_t k,uint32_t limit,bool indexed,bool runs){
    if(!batch || batch>65535)throw std::invalid_argument("batch capacity must be in [1,65535]");
    Options o;o.sort_limit=limit;validate(n,k,o);
    BatchArenaLayout result;ArenaPlanner p;
    result.descriptor_slots=batch_descriptor_slots(k,limit);
    uint64_t b=batch,m=std::max(k,limit);
    std::array<uint64_t,6> counts{{b*n,b*n,b*k,b*m,b*m,runs?b*k:0}};
    for(size_t j=0;j<counts.size();++j){
        result.values[j]=p.add(counts[j],sizeof(Value));
        result.indices[j]=p.add(indexed?counts[j]:0,sizeof(uint32_t));
    }
    result.histogram=p.add(b*4097,sizeof(uint32_t));
    result.metadata=p.add(b,sizeof(Decision));result.counters=p.add(b*2,sizeof(uint32_t));
    result.jobs=p.add(b*result.descriptor_slots,sizeof(KernelJobT<Value>));
    result.bytes=p.finish();return result;
}
inline void validate_arena(void* ptr,uint64_t bytes,const BatchArenaLayout& layout){
    auto address=reinterpret_cast<uintptr_t>(ptr);
    if(!ptr || (address&(ArenaPlanner::alignment-1)) || bytes<layout.bytes ||
       layout.bytes>std::numeric_limits<uintptr_t>::max()-address)
        throw std::invalid_argument("workspace arena must be 256-byte aligned and sufficiently large");
}
} // namespace detail
} // namespace r2sk
