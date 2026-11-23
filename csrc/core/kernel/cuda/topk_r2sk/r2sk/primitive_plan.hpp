#pragma once
#include "core.hpp"

namespace r2sk {
// Scalar peel + aligned float4 interior + scalar tail. Never round the input
// pointer down: the caller need not own any values before device_input.
struct PackPlan {
    uint32_t peel,packs,tail;
    R2SK_HD uint32_t tail_begin() const {return peel+4*packs;}
};
template<class Value> R2SK_HD inline PackPlan value_pack_plan(uint64_t address,uint32_t n){
    constexpr uint32_t alignment=4*sizeof(Value);
    uint32_t peel=uint32_t((alignment-(address&(alignment-1)))&(alignment-1))/sizeof(Value);
    if(peel>n)peel=n;
    uint32_t packs=(n-peel)/4;
    return {peel,packs,n-peel-4*packs};
}
R2SK_HD inline PackPlan pack_plan(uint64_t address,uint32_t n){return value_pack_plan<float>(address,n);}
constexpr uint32_t packed_tile_values=256*4;
R2SK_HD inline bool flush_cache(uint32_t count,uint32_t capacity,uint32_t next_tile,bool force){
    return count && (force || capacity<next_tile || count>capacity-next_tile);
}
inline uint32_t gather_shared_bytes(uint32_t gcap,bool indexed,bool buffered){
    uint32_t capacity=packed_tile_values*(buffered?2u:1u);
    uint32_t gslots=std::min(gcap,capacity);
    return (gslots+capacity)*4*(indexed?2u:1u);
}
} // namespace r2sk
