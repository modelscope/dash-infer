// Optimized primitive paths adapted from the ideas in RadiK's countBinEx,
// selectCandidateEx, selectBin and warpPrefixSum, pinned at
// c934b15d45655bec1cf4fdc4fcb04ba4f31511cf.
// Copyright (c) 2023-present Alibaba Inc. MIT license: LICENSE.RadiK.
// Rewritten here: bounded dual destinations, carried indices, safe scalar
// alignment peel/tail, two-level warp scan and independently selectable modes.
// Included inside r2sk's anonymous namespace, after KernelJobT<Value> is declared.

// Keep four logical values per lane in both storage modes. BF16 packs are
// 8-byte aligned uint2 reads; FP32 keeps the original 16-byte float4 read.
struct FourValues {float values[4];};
template<class Value> __device__ FourValues load_four(const Value* input){
    if constexpr(std::is_same<Value,float>::value){
        float4 x=*reinterpret_cast<const float4*>(input);return {{x.x,x.y,x.z,x.w}};
    }else{
        static_assert(std::is_same<Value,BFloat16>::value,"supported storage type");
        uint2 x=*reinterpret_cast<const uint2*>(input);
        return {{__uint_as_float((x.x&0xffffu)<<16),__uint_as_float(x.x&0xffff0000u),
                 __uint_as_float((x.y&0xffffu)<<16),__uint_as_float(x.y&0xffff0000u)}};
    }
}

template<int BINS,int THREADS,bool BATCHED=false,class Value=float>
__global__ void __launch_bounds__(THREADS) count_packed(ValueView<Value> src,uint32_t n,int pass,uint32_t* hist,const KernelJobT<Value>* jobs=nullptr,uint32_t* scale_unsafe=nullptr){
    if constexpr(BATCHED){auto job=jobs[blockIdx.y];src=job.src;n=job.n;pass=job.pass;hist=job.hist;scale_unsafe=job.scale_unsafe;}
    __shared__ uint32_t local[BINS];
    float reference=src.shift_reference && pass<3?*src.shift_reference:0.f;
    for(int j=threadIdx.x;j<BINS;j+=THREADS)local[j]=0;
    __syncthreads();
    bool shifted=src.shift_reference && pass<3;int unsafe=0;
    auto update=[&](float value){float key_value=shifted?shift_value(value,reference):value;
        atomicAdd(local+digit(key_value,pass),1u);
        if(scale_unsafe && (shifted ? !shift_reversible(value,key_value,reference) : !finite_key(value)))unsafe=1;};
    auto plan=value_pack_plan<Value>(reinterpret_cast<uintptr_t>(src.v),n);
    const Value* input=src.v+plan.peel;
    for(uint64_t q=uint64_t(blockIdx.x)*THREADS+threadIdx.x;q<plan.packs;q+=uint64_t(gridDim.x)*THREADS){
        auto packed=load_four(input+4*q);const float* values=packed.values;
        #pragma unroll
        for(int j=0;j<4;++j)update(values[j]);
    }
    if(blockIdx.x==0 && threadIdx.x<plan.peel+plan.tail){
        uint32_t t=threadIdx.x;
        uint32_t j=t<plan.peel?t:plan.tail_begin()+t-plan.peel;
        update(src.v[j]);
    }
    if(scale_unsafe){int bad=__syncthreads_or(unsafe);if(threadIdx.x==0 && bad)atomicOr(scale_unsafe,1u);}
    else __syncthreads();
    for(int j=threadIdx.x;j<BINS;j+=THREADS)if(local[j])atomicAdd(hist+j,local[j]);
}

template<int BINS,bool BATCHED=false,bool CAPTURE=false,class Value=float>
__global__ void select_bin_warp(const uint32_t* hist,uint32_t rank,Decision* d,const KernelJobT<Value>* jobs=nullptr,
                               const uint32_t* captured_hist=nullptr,const uint32_t* captured_count=nullptr,const uint32_t* scale_unsafe=nullptr){
    constexpr int THREADS=BINS==4096?1024:256;
    constexpr int ITEMS=BINS/THREADS,WARPS=THREADS/32;
    if constexpr(BATCHED){auto job=jobs[blockIdx.y];hist=job.hist;rank=job.rank;d=job.decision;scale_unsafe=job.scale_unsafe;}
    __shared__ uint32_t warp_prefix[WARPS];
    uint32_t counts[ITEMS],own=0;
    #pragma unroll
    for(int j=0;j<ITEMS;++j){counts[j]=hist[BINS-1-(threadIdx.x*ITEMS+j)];own+=counts[j];}
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    uint32_t inclusive=own;
    #pragma unroll
    for(int delta=1;delta<32;delta*=2){uint32_t other=__shfl_up_sync(0xffffffffu,inclusive,delta);
        if(lane>=delta)inclusive+=other;}
    if(lane==31)warp_prefix[warp]=inclusive;
    __syncthreads();
    if(warp==0){
        uint32_t value=lane<WARPS?warp_prefix[lane]:0;
        #pragma unroll
        for(int delta=1;delta<32;delta*=2){uint32_t other=__shfl_up_sync(0xffffffffu,value,delta);
            if(lane>=delta)value+=other;}
        if(lane<WARPS)warp_prefix[lane]=value;
    }
    __syncthreads();
    uint32_t before=inclusive-own+(warp?warp_prefix[warp-1]:0);
    if(threadIdx.x==0){d->total=warp_prefix[WARPS-1];
        d->scale_unsafe=scale_unsafe?*scale_unsafe:0;
        if constexpr(!CAPTURE){d->captured=0;d->captured_retained=0;}}
    if(before<rank && rank<=before+own){
        #pragma unroll
        for(int j=0;j<ITEMS;++j){
            if(rank<=before+counts[j]){d->bin=BINS-1-(threadIdx.x*ITEMS+j);d->h=before;d->B=counts[j];break;}
            before+=counts[j];
        }
    }
    if constexpr(CAPTURE){
        __syncthreads();
        if(threadIdx.x==0){uint32_t retained=0;
            for(int bin=d->bin;bin<BINS;++bin)retained+=captured_hist[bin];
            d->captured=*captured_count;d->captured_retained=retained;}
    }
}

template<bool INDEXED,bool FILTER,bool BUFFERED,class Value>
struct GatherCache {
    static constexpr uint32_t CAPACITY=packed_tile_values*(BUFFERED?2:1);
    ValueView<Value> src;ValueBuffer<Value> G,E;uint32_t boundary,pass,gcap,ecap,gslots;float reference;
    float* gv;float* ev;uint32_t* gi;uint32_t* ei;
    uint32_t *gc,*ec,*gb,*eb,*counters;
    __device__ void stage(float value,uint32_t j){
        uint32_t d=FILTER?ordered_key(value):digit(value,pass,reference,src.shift_reference!=nullptr);
        if(d>boundary && G.v){
            uint32_t pos=atomicAdd(gc,1u);gv[pos]=value;
            if constexpr(INDEXED)gi[pos]=src.i?src.i[j]:j;
        }else if(d==boundary){
            uint32_t pos=atomicAdd(ec,1u);ev[pos]=value;
            if constexpr(INDEXED)ei[pos]=src.i?src.i[j]:j;
        }
    }
    // Every thread executes every barrier, even in partial vector tiles.
    __device__ void flush(bool force){
        __syncthreads();
        uint32_t ng=*gc,ne=*ec;
        bool fg=flush_cache(ng,gslots,packed_tile_values,force || !BUFFERED);
        bool fe=flush_cache(ne,CAPACITY,packed_tile_values,force || !BUFFERED);
        if(threadIdx.x==0){*gb=fg?atomicAdd(counters,ng):0;*eb=fe?atomicAdd(counters+1,ne):0;}
        __syncthreads();
        if(fg)for(uint32_t t=threadIdx.x;t<ng;t+=BLOCK)if(*gb+t<gcap){
            G.v[*gb+t]=gv[t];if constexpr(INDEXED)G.i[*gb+t]=gi[t];}
        if(fe)for(uint32_t t=threadIdx.x;t<ne;t+=BLOCK)if(*eb+t<ecap){
            E.v[*eb+t]=ev[t];if constexpr(INDEXED)E.i[*eb+t]=ei[t];}
        __syncthreads();
        if(threadIdx.x==0){if(fg)*gc=0;if(fe)*ec=0;}
        __syncthreads();
    }
};

template<bool INDEXED,bool FILTER,bool BUFFERED,bool BATCHED=false,class Value=float>
__global__ void __launch_bounds__(BLOCK) gather_packed(ValueView<Value> src,uint32_t n,int pass,uint32_t boundary,
                             ValueBuffer<Value> G,uint32_t gcap,ValueBuffer<Value> E,uint32_t ecap,uint32_t* counters,const KernelJobT<Value>* jobs=nullptr){
    if constexpr(BATCHED){auto job=jobs[blockIdx.y];src=job.src;n=job.n;pass=job.pass;
        boundary=FILTER && job.pivot.v?ordered_key(job.pivot.v[0]):job.boundary;
        G=job.G;gcap=job.gcap;E=job.E;ecap=job.ecap;counters=job.counters;}
    using Cache=GatherCache<INDEXED,FILTER,BUFFERED,Value>;
    uint32_t gslots=gcap<Cache::CAPACITY?gcap:Cache::CAPACITY;
    extern __shared__ uint32_t storage[];
    float* gv=reinterpret_cast<float*>(storage);float* ev=gv+gslots;
    uint32_t* gi=storage+gslots+Cache::CAPACITY;
    uint32_t* ei=gi+(INDEXED?gslots:0);
    __shared__ uint32_t gc,ec,gb,eb;
    float reference=src.shift_reference && pass<3?*src.shift_reference:0.f;
    if(threadIdx.x==0){gc=0;ec=0;}
    __syncthreads();
    Cache cache{src,G,E,boundary,uint32_t(pass),gcap,ecap,gslots,reference,gv,ev,gi,ei,&gc,&ec,&gb,&eb,counters};
    auto plan=value_pack_plan<Value>(reinterpret_cast<uintptr_t>(src.v),n);
    const Value* input=src.v+plan.peel;
    for(uint64_t base=uint64_t(blockIdx.x)*BLOCK;base<plan.packs;base+=uint64_t(gridDim.x)*BLOCK){
        uint64_t q=base+threadIdx.x;
        if(q<plan.packs){
            auto packed=load_four(input+4*q);const float* values=packed.values;
            #pragma unroll
            for(int t=0;t<4;++t)cache.stage(values[t],plan.peel+uint32_t(q)*4+t);
        }
        cache.flush(false);
    }
    if(blockIdx.x==0 && threadIdx.x<plan.peel+plan.tail){
        uint32_t t=threadIdx.x,j=t<plan.peel?t:plan.tail_begin()+t-plan.peel;
        cache.stage(src.v[j],j);
    }
    cache.flush(true);
}
