#define CL_TARGET_OPENCL_VERSION 120
#include "xtg/opencl.h"

#include <CL/cl.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>

#include "xtg/common.h"

namespace xtg {
namespace {
void check(cl_int e, const char* op) { XTG_CHECK(e == CL_SUCCESS, "OpenCL %s failed: %d", op, e); }
const char* source = R"CLC(
#ifdef cl_khr_subgroup_shuffle_relative
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable
#endif
__constant float e2[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
float fp8(uchar b) {
    uint a=b&127, ex=a>>3, ma=a&7;
    float v=ex?as_float((ex+120)<<23)*(1.0f+ma*.125f):ma*.001953125f;
    return b&128?-v:v;
}
float4 unpack4(ushort b){
    uint4 code=(uint4)(b&15,(b>>4)&15,(b>>8)&15,b>>12),mag=code&7,ex=mag>>1;
    float4 v=as_float4(((ex+126)<<23)|((mag&1)<<22));
    v=select(v,convert_float4(mag)*.5f,mag<2);
    return as_float4(as_uint4(v)|((code&8)<<28));
}
// NVFP4 on the device is k-major: codes [K/32][N] in 16-byte blocks (the 32 codes of one row, low nibble first),
// scales [K/32][N] as two E4M3 bytes (the low one for the first 16). The work-items of a wave take one row each,
// so they read consecutive blocks while the activations are one address for all of them.
float weight4(__global const uchar* c,__global const uchar* s,int row,int k,int K,int N){
    size_t blk=(size_t)(k/32)*N+row;
    uchar v=c[blk*16+(k%32)/2];
    return e2[(k&1)?v>>4:v&15]*fp8(s[blk*2+(k%32)/16]);
}
// x . 32 weights of block b: two 16-weight groups, each summed and then scaled
float block32(__global const uchar*c,__global const uchar*s,size_t blk,__global const float*xb){
    uint4 q=vload4(0,(__global const uint*)(c+blk*16));
    ushort sv=((__global const ushort*)s)[blk];
    float4 g0=unpack4(q.x)*vload4(0,xb),g1=unpack4(q.z)*vload4(4,xb);
    g0=mad(unpack4(q.x>>16),vload4(1,xb),g0);g0=mad(unpack4(q.y),vload4(2,xb),g0);g0=mad(unpack4(q.y>>16),vload4(3,xb),g0);
    g1=mad(unpack4(q.z>>16),vload4(5,xb),g1);g1=mad(unpack4(q.w),vload4(6,xb),g1);g1=mad(unpack4(q.w>>16),vload4(7,xb),g1);
    return mad(fp8(sv&255),g0.x+g0.y+g0.z+g0.w,fp8(sv>>8)*(g1.x+g1.y+g1.z+g1.w));
}
// the same through an image1d_buffer view of the codes (RGBA32UI, one block per texel): faster on Adreno 830 for
// the wide K = 2560 projections (qkvg, gate / up: 47-50 against 41-43 GB/s), slower for o and down
float block32i(__read_only image1d_buffer_t c,__global const uchar*s,size_t blk,__global const float*xb){
    uint4 q=read_imageui(c,(int)blk);
    ushort sv=((__global const ushort*)s)[blk];
    float4 g0=unpack4(q.x)*vload4(0,xb),g1=unpack4(q.z)*vload4(4,xb);
    g0=mad(unpack4(q.x>>16),vload4(1,xb),g0);g0=mad(unpack4(q.y),vload4(2,xb),g0);g0=mad(unpack4(q.y>>16),vload4(3,xb),g0);
    g1=mad(unpack4(q.z>>16),vload4(5,xb),g1);g1=mad(unpack4(q.w),vload4(6,xb),g1);g1=mad(unpack4(q.w>>16),vload4(7,xb),g1);
    return mad(fp8(sv&255),g0.x+g0.y+g0.z+g0.w,fp8(sv>>8)*(g1.x+g1.y+g1.z+g1.w));
}
// Decode GEMV for up to four fused projections: one output row per work-item, the KS = get_local_size(1)
// work-items of a row take a share of K each and add up in local memory (local size (256 / KS, KS) at most).
// The row's projection: counts are the rows of each, from the start of the fused output.
#define FUSED_ROW(counts) \
    int outrow=get_global_id(0),lr=get_local_id(0),ks=get_local_id(1),KS=get_local_size(1); \
    int row=outrow,N=counts.x+counts.y+counts.z+counts.w,n=counts.x,j=0; \
    if(row>=counts.x){row-=counts.x;n=counts.y;j=1; \
        if(row>=counts.y){row-=counts.y;n=counts.z;j=2; \
            if(row>=counts.z){row-=counts.z;n=counts.w;j=3;}}} \
    int nb=K/32,per=(nb+KS-1)/KS,b0=ks*per,b1=min(nb,b0+per)
float ksum(float acc,int ks,int lr,int KS,__local float*red){   // the KS shares of a row, valid for ks == 0
    red[ks*64+lr]=acc;barrier(CLK_LOCAL_MEM_FENCE);
    if(ks==0)for(int i=1;i<KS;i++)acc+=red[i*64+lr];
    barrier(CLK_LOCAL_MEM_FENCE);
    return acc;
}
__kernel void gemv4(__global const uchar*c0,__global const uchar*s0,
                   __global const uchar*c1,__global const uchar*s1,
                   __global const uchar*c2,__global const uchar*s2,
                   __global const uchar*c3,__global const uchar*s3,
                   __global const float*x,__global float*y,int K,int4 counts,float4 scales,int off){
    FUSED_ROW(counts);
    __global const uchar*c=j==0?c0:j==1?c1:j==2?c2:c3;__global const uchar*s=j==0?s0:j==1?s1:j==2?s2:s3;
    __local float red[8*64];
    float acc=0;
    if(outrow<N)for(int b=b0;b<b1;b++)acc+=block32(c,s,(size_t)b*n+row,x+b*32);
    acc=ksum(acc,ks,lr,KS,red);
    if(ks==0&&outrow<N)y[off+outrow]=acc*(j==0?scales.x:j==1?scales.y:j==2?scales.z:scales.w);
}
__kernel void gemv4i(__read_only image1d_buffer_t c0,__global const uchar*s0,
                    __read_only image1d_buffer_t c1,__global const uchar*s1,
                    __read_only image1d_buffer_t c2,__global const uchar*s2,
                    __read_only image1d_buffer_t c3,__global const uchar*s3,
                    __global const float*x,__global float*y,int K,int4 counts,float4 scales,int off){
    FUSED_ROW(counts);
    __global const uchar*s=j==0?s0:j==1?s1:j==2?s2:s3;
    __local float red[8*64];
    float acc=0;
    if(outrow<N)for(int b=b0;b<b1;b++){   // images cannot be chosen through a variable: one call per projection
        size_t blk=(size_t)b*n+row;
        acc+=j==0?block32i(c0,s,blk,x+b*32):j==1?block32i(c1,s,blk,x+b*32):j==2?block32i(c2,s,blk,x+b*32):block32i(c3,s,blk,x+b*32);
    }
    acc=ksum(acc,ks,lr,KS,red);
    if(ks==0&&outrow<N)y[off+outrow]=acc*(j==0?scales.x:j==1?scales.y:j==2?scales.z:scales.w);
}
__kernel void gemv16(__global const ushort*w,__global const float*x,__global float*y,int K,int N,int offset){
    int lid=get_local_id(0),lane=lid&31,row=get_group_id(0)*4+lid/32;float4 acc=0;
    if(row<N)for(int k=lane*4;k<K;k+=128){
        float4 v=as_float4(convert_uint4(vload4(0,w+(size_t)row*K+k))<<16);
        acc=mad(v,vload4(0,x+k),acc);
    }
    float sum=acc.x+acc.y+acc.z+acc.w;
#ifdef cl_khr_subgroup_shuffle_relative
    if(get_sub_group_size()>=32){
        for(int step=16;step;step/=2)sum+=sub_group_shuffle_xor(sum,step);
        if(lane==0&&row<N)y[offset+row]=sum;return;
    }
#endif
    __local float sums[128];sums[lid]=sum;barrier(CLK_LOCAL_MEM_FENCE);
    for(int step=16;step;step/=2){if(lane<step)sums[lid]+=sums[lid+step];barrier(CLK_LOCAL_MEM_FENCE);}
    if(lane==0&&row<N)y[offset+row]=sums[lid];
}
// 2..MRMAX activation rows at once (verifying speculative drafts): each block of weights is read and decoded once
// and used for every row, so the call costs about what one row does. Same layout and work split as gemv4; the
// outputs of a projection are an [m][n] block (gemm4's layout). Loops over rows are unrolled so the accumulators
// stay registers.
#define MRMAX 16
__kernel void gemvm(__global const uchar*c0,__global const uchar*s0,
                    __global const uchar*c1,__global const uchar*s1,
                    __global const uchar*c2,__global const uchar*s2,
                    __global const uchar*c3,__global const uchar*s3,
                    __global const float*x,__global float*y,int K,int4 counts,float4 scales,int off,int m){
    FUSED_ROW(counts);
    __global const uchar*c=j==0?c0:j==1?c1:j==2?c2:c3;__global const uchar*s=j==0?s0:j==1?s1:j==2?s2:s3;
    const float scale=j==0?scales.x:j==1?scales.y:j==2?scales.z:scales.w;
    const int base=outrow-row;   // the first row of this projection in the fused output
    float acc[MRMAX];
#pragma unroll
    for(int r=0;r<MRMAX;r++)acc[r]=0;
    if(outrow<N)for(int b=b0;b<b1;b++){
        size_t blk=(size_t)b*n+row;
        uint4 q=vload4(0,(__global const uint*)(c+blk*16));
        ushort sv=((__global const ushort*)s)[blk];
        float s0=fp8(sv&255),s1=fp8(sv>>8);
        float4 w0=unpack4(q.x),w1=unpack4(q.x>>16),w2=unpack4(q.y),w3=unpack4(q.y>>16);
        float4 w4=unpack4(q.z),w5=unpack4(q.z>>16),w6=unpack4(q.w),w7=unpack4(q.w>>16);
#pragma unroll
        for(int r=0;r<MRMAX;r++)if(r<m){
            __global const float*xr=x+(size_t)r*K+b*32;
            float4 g0=w0*vload4(0,xr),g1=w4*vload4(4,xr);
            g0=mad(w1,vload4(1,xr),g0);g0=mad(w2,vload4(2,xr),g0);g0=mad(w3,vload4(3,xr),g0);
            g1=mad(w5,vload4(5,xr),g1);g1=mad(w6,vload4(6,xr),g1);g1=mad(w7,vload4(7,xr),g1);
            acc[r]+=mad(s0,g0.x+g0.y+g0.z+g0.w,s1*(g1.x+g1.y+g1.z+g1.w));   // as block32: a verify row = a decode row
        }
    }
    __local float red[8*64];
#pragma unroll
    for(int r=0;r<MRMAX;r++)if(r<m){
        float t=ksum(acc[r],ks,lr,KS,red);
        if(ks==0&&outrow<N)y[off+(size_t)base*m+(size_t)r*n+row]=t*scale;
    }
}
// the BF16 head keeps its row-major layout: 32 lanes per row, summed in local memory
float lanes32(float v,__local float*sums,int lid,int lane){
    sums[lid]=v;barrier(CLK_LOCAL_MEM_FENCE);
    for(int step=16;step;step/=2){if(lane<step)sums[lid]+=sums[lid+step];barrier(CLK_LOCAL_MEM_FENCE);}
    float r=sums[lid-lane];barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
__kernel void gemv16m(__global const ushort*w,__global const float*x,__global float*y,int K,int N,int m){
    int lid=get_local_id(0),lane=lid&31,row=get_group_id(0)*4+lid/32;
    float4 acc[MRMAX];
#pragma unroll
    for(int r=0;r<MRMAX;r++)acc[r]=0;
    if(row<N)for(int k=lane*4;k<K;k+=128){
        float4 v=as_float4(convert_uint4(vload4(0,w+(size_t)row*K+k))<<16);
#pragma unroll
        for(int r=0;r<MRMAX;r++)if(r<m)acc[r]=mad(v,vload4(0,x+(size_t)r*K+k),acc[r]);
    }
    __local float sums[128];
#pragma unroll
    for(int r=0;r<MRMAX;r++)if(r<m){
        float sum=lanes32(acc[r].x+acc[r].y+acc[r].z+acc[r].w,sums,lid,lane);
        if(lane==0&&row<N)y[(size_t)r*N+row]=sum;
    }
}
// 64x64 output tile, K=32, 4x4 FP32 accumulators per thread. No activation quantization.
__kernel void gemm4(__global const uchar*c,__global const uchar*s,__global const float*x,
                   __global float*y,int K,int N,int M,float scale,int offset){
    int tx=get_local_id(0),ty=get_local_id(1),tid=ty*16+tx;
    int row=get_group_id(1)*64+ty*4,col=get_group_id(0)*64+tx*4;
    __local float aa[64][32],bb[32][64];float4 r0=0,r1=0,r2=0,r3=0;
    for(int base=0;base<K;base+=32){
        for(int i=tid;i<2048;i+=256){
            int ar=i/32,ak=i%32,gr=get_group_id(1)*64+ar;
            aa[ar][ak]=(gr<M&&base+ak<K)?x[(size_t)gr*K+base+ak]:0;
            int bk=i/64,bn=i%64,gn=get_group_id(0)*64+bn;
            bb[bk][bn]=(gn<N&&base+bk<K)?weight4(c,s,gn,base+bk,K,N):0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<32;k++){
            float4 b=vload4(0,&bb[k][tx*4]);
            r0=mad((float4)(aa[ty*4][k]),b,r0);r1=mad((float4)(aa[ty*4+1][k]),b,r1);
            r2=mad((float4)(aa[ty*4+2][k]),b,r2);r3=mad((float4)(aa[ty*4+3][k]),b,r3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int i=0;i<4;i++){float4 r=i==0?r0:i==1?r1:i==2?r2:r3;
        if(row+i<M){size_t dst=offset+(size_t)(row+i)*N+col; if(col<N)y[dst]=r.x*scale; if(col+1<N)y[dst+1]=r.y*scale; if(col+2<N)y[dst+2]=r.z*scale; if(col+3<N)y[dst+3]=r.w*scale;}}
}
__kernel void gemm16(__global const ushort*w,__global const float*x,__global float*y,int K,int N,int M,int offset){
    int tx=get_local_id(0),ty=get_local_id(1),tid=ty*16+tx;
    int row=get_group_id(1)*64+ty*4,col=get_group_id(0)*64+tx*4;
    __local float aa[64][32],bb[32][64];float4 r0=0,r1=0,r2=0,r3=0;
    for(int base=0;base<K;base+=32){
        for(int i=tid;i<2048;i+=256){
            int ar=i/32,ak=i%32,gr=get_group_id(1)*64+ar;
            aa[ar][ak]=(gr<M&&base+ak<K)?x[(size_t)gr*K+base+ak]:0;
            int bk=i/64,bn=i%64,gn=get_group_id(0)*64+bn;
            bb[bk][bn]=(gn<N&&base+bk<K)?as_float((uint)w[(size_t)gn*K+base+bk]<<16):0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k=0;k<32;k++){float4 b=vload4(0,&bb[k][tx*4]);
            r0=mad((float4)(aa[ty*4][k]),b,r0);r1=mad((float4)(aa[ty*4+1][k]),b,r1);
            r2=mad((float4)(aa[ty*4+2][k]),b,r2);r3=mad((float4)(aa[ty*4+3][k]),b,r3);}
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int i=0;i<4;i++){float4 r=i==0?r0:i==1?r1:i==2?r2:r3;
        if(row+i<M){size_t dst=offset+(size_t)(row+i)*N+col; if(col<N)y[dst]=r.x; if(col+1<N)y[dst+1]=r.y; if(col+2<N)y[dst+2]=r.z; if(col+3<N)y[dst+3]=r.w;}}
}
// The prefill GEMM in T precision (half where the device has cl_khr_fp16, else float): a 64 x 128 tile per 256
// work-items, each 4 rows x 8 columns; the FP32 activations become T on their way into local memory and NVFP4 is
// decoded once per tile; T products summed per 32-k step, those sums added in FP32. N must be a multiple of 128.
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half T; typedef half8 T8;
#define TO_T8 convert_half8
#else
typedef float T; typedef float8 T8;
#define TO_T8 convert_float8
#endif
__kernel void gemmt(__global const uchar*c,__global const uchar*s,__global const float*A,__global float*C,
                    int K,int N,int M,float scale,int offset){
    __local T As[64][32+2];
    __local T Bs[32][128];
    const int tid=get_local_id(0),tm=tid/16,tn=tid%16,m0=get_group_id(1)*64,n0=get_group_id(0)*128;
    float8 acc0=0,acc1=0,acc2=0,acc3=0;
    for(int k0=0;k0<K;k0+=32){
        {int r=tid/4,kk=(tid%4)*8;
         float8 a=m0+r<M?vload8(0,A+(size_t)(m0+r)*K+k0+kk):(float8)0;
         vstore8(TO_T8(a),0,&As[r][kk]);}
        {int r=tid/2,h=(tid%2)*16,n=n0+r;
         size_t blk=(size_t)(k0/32)*N+n;   // the two work-items of a row read the halves of its block
         uint2 q=vload2(0,(__global const uint*)(c+blk*16+h/2));
         ushort sv=((__global const ushort*)s)[blk];
         float sc=fp8(h?sv>>8:sv&255);
         float4 w0=unpack4(q.x)*sc,w1=unpack4(q.x>>16)*sc,w2=unpack4(q.y)*sc,w3=unpack4(q.y>>16)*sc;
         float wv[16]={w0.x,w0.y,w0.z,w0.w,w1.x,w1.y,w1.z,w1.w,w2.x,w2.y,w2.z,w2.w,w3.x,w3.y,w3.z,w3.w};
         for(int i=0;i<16;i++)Bs[h+i][r]=(T)wv[i];}
        barrier(CLK_LOCAL_MEM_FENCE);
        T8 p0=0,p1=0,p2=0,p3=0;
        for(int k=0;k<32;k++){
            T8 b=vload8(0,&Bs[k][tn*8]);
            p0=mad((T8)As[tm*4][k],b,p0);p1=mad((T8)As[tm*4+1][k],b,p1);
            p2=mad((T8)As[tm*4+2][k],b,p2);p3=mad((T8)As[tm*4+3][k],b,p3);
        }
        acc0+=convert_float8(p0);acc1+=convert_float8(p1);acc2+=convert_float8(p2);acc3+=convert_float8(p3);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float8 acc[4]={acc0,acc1,acc2,acc3};
    for(int i=0;i<4;i++){int m=m0+tm*4+i;
        if(m<M)vstore8(acc[i]*scale,0,C+offset+(size_t)m*N+n0+tn*8);}
}
)CLC";
}  // namespace

struct OpenClBackend::Impl {
#define API(X)                                                                                                 \
  X(clGetPlatformIDs)                                                                                          \
  X(clGetDeviceIDs)                                                                                            \
  X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue) X(clCreateProgramWithSource) X(clBuildProgram) \
      X(clGetProgramBuildInfo) X(clCreateKernel) X(clCreateBuffer) X(clSetKernelArg) X(clEnqueueNDRangeKernel) \
          X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clFinish) X(clReleaseMemObject) X(clReleaseKernel)  \
              X(clReleaseProgram) X(clReleaseCommandQueue) X(clReleaseContext) X(clEnqueueCopyBuffer) \
                  X(clGetEventProfilingInfo) X(clReleaseEvent) X(clGetKernelInfo) X(clCreateImage)
#define DECL(n) decltype(&::n) n = nullptr;
  API(DECL)
#undef DECL
  // the device-resident path's programs (by source and options) and kernels (by program and name)
  std::map<std::pair<const char*, std::string>, cl_program> programs;
  std::map<std::pair<cl_program, std::string>, cl_kernel> kernels;
  // XTG_GPU_PROFILE: every kernel gets an event; finish() adds their device times up per kernel name
  bool profiling = false;
  std::vector<std::pair<cl_kernel, cl_event>> events;
  std::map<std::string, std::pair<double, size_t>> kernel_time;   // name -> seconds, launches
  cl_event* event_for(cl_kernel k) {
    if (!profiling) return nullptr;
    events.push_back({k, nullptr});
    return &events.back().second;
  }
  void collect() {
    for (auto& [k, ev] : events) {
      cl_ulong a = 0, b = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr);
      char n[64] = {};
      clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(n) - 1, n, nullptr);
      auto& t = kernel_time[n];
      t.first += double(b - a) * 1e-9;
      t.second++;
      clReleaseEvent(ev);
    }
    events.clear();
  }
  void* lib = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel gemv4 = nullptr, gemv4i = nullptr, gemm4 = nullptr, gemv16 = nullptr, gemm16 = nullptr, gemvm = nullptr,
            gemv16m = nullptr, gemmt = nullptr;
  size_t image_max = 0;   // texels of an image1d_buffer
  bool fp16 = false, use_gemmt = true;   // the device has cl_khr_fp16; XTG_GEMM=fp32 keeps gemm4 for prefill
  cl_mem input = nullptr, output = nullptr;
  size_t input_size = 0, output_size = 0, weight_size = 0;
  double decode_nv_s = 0, decode_bf_s = 0;
  size_t decode_nv_calls = 0, decode_bf_calls = 0;
  std::string name;
  struct Weight {
    cl_mem codes = nullptr, scales = nullptr, image = nullptr;   // image: an RGBA32UI view of the codes
  };
  std::unordered_map<const void*, Weight> weights;

  ~Impl() {
    if (std::getenv("XTG_GPU_PROFILE"))
      log("GPU decode: NVFP4 %.6fs/%zu calls, BF16 %.6fs/%zu calls", decode_nv_s, decode_nv_calls, decode_bf_s,
          decode_bf_calls);
    if (queue) clFinish(queue);
    if (profiling) {
      collect();
      double total = 0;
      for (auto& [n, t] : kernel_time) total += t.first;
      for (auto& [n, t] : kernel_time)
        log("GPU kernel %-12s %9.3f ms %7zu launches %5.1f%%", n.c_str(), t.first * 1e3, t.second,
            total > 0 ? 100 * t.first / total : 0.0);
      if (total > 0) log("GPU kernels total %.3f ms", total * 1e3);
    }
    for (auto& [key, w] : weights) {
      if (w.image) clReleaseMemObject(w.image);
      if (w.codes) clReleaseMemObject(w.codes);
      if (w.scales) clReleaseMemObject(w.scales);
    }
    if (input) clReleaseMemObject(input);
    if (output) clReleaseMemObject(output);
    for (auto k : {gemv4, gemv4i, gemm4, gemv16, gemm16, gemvm, gemv16m, gemmt})
      if (k) clReleaseKernel(k);
    for (auto& [key, k] : kernels) clReleaseKernel(k);
    for (auto& [key, p] : programs) clReleaseProgram(p);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    if (lib) dlclose(lib);
  }
  void init() {
    lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    XTG_CHECK(lib, "OpenCL driver unavailable: %s", dlerror());
#define LOAD(n)                                      \
  n = reinterpret_cast<decltype(n)>(dlsym(lib, #n)); \
  XTG_CHECK(n, "OpenCL missing " #n);
    API(LOAD)
#undef LOAD
#undef API
    cl_platform_id ps[8];
    cl_uint count = 0;
    check(clGetPlatformIDs(8, ps, &count), "platforms");
    for (cl_uint i = 0; i < std::min(count, 8u); i++)
      if (clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
    XTG_CHECK(device, "No OpenCL GPU found");
    char text[256] = {};
    check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(text), text, nullptr), "device name");
    name = text;
    cl_int e;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &e);
    check(e, "context");
    // Optional Qualcomm context-local hint; does not change global device settings.
    using SetPerfHint = cl_int (*)(cl_context, cl_uint);
    auto perf = reinterpret_cast<SetPerfHint>(dlsym(lib, "clSetPerfHintQCOM"));
    if (perf) log("OpenCL high performance hint: %d", perf(context, 0x40C3));
    profiling = std::getenv("XTG_GPU_PROFILE") != nullptr;
    queue = clCreateCommandQueue(context, device, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &e);
    check(e, "queue");
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &e);
    check(e, "program");
    e = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (e) {
      char log[16384] = {};
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
      fail("OpenCL build: %s", log);
    }
    auto create = [&](const char* n) {
      cl_kernel k = clCreateKernel(program, n, &e);
      check(e, n);
      return k;
    };
    gemv4 = create("gemv4");
    gemv4i = create("gemv4i");
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(image_max), &image_max, nullptr);
    gemm4 = create("gemm4");
    gemv16 = create("gemv16");
    gemm16 = create("gemm16");
    gemvm = create("gemvm");
    gemv16m = create("gemv16m");
    gemmt = create("gemmt");
    char ext[8192] = {};
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, nullptr);
    fp16 = std::strstr(ext, "cl_khr_fp16") != nullptr;
    const char* gm = std::getenv("XTG_GEMM");
    use_gemmt = !(gm && std::string(gm) == "fp32");
    no_image = std::getenv("XTG_GEMV_NOIMAGE") != nullptr;
    if (const char* v = std::getenv("XTG_GEMV_KS")) {
      gemv_ks = std::atoi(v);
      XTG_CHECK(gemv_ks == 1 || gemv_ks == 2 || gemv_ks == 4 || gemv_ks == 8, "XTG_GEMV_KS must be 1, 2, 4 or 8");
    }
    log("OpenCL device: %s (FP32 accumulation, packed weights; prefill GEMM %s)", name.c_str(),
        use_gemmt ? (fp16 ? "FP16 tiles" : "FP32 tiles") : "gemm4");
  }
  cl_mem copy(const void* data, size_t bytes) {
    cl_int e;
    cl_mem p = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(data), &e);
    check(e, "weight allocation");
    weight_size += bytes;
    return p;
  }
  void scratch(cl_mem& b, size_t& size, size_t need) {
    if (size >= need) return;
    if (b) {
      check(clReleaseMemObject(b), "release scratch");
      b = nullptr;
      size = 0;
    }
    cl_int e;
    b = clCreateBuffer(context, CL_MEM_READ_WRITE, need, nullptr, &e);
    check(e, "scratch allocation");
    size = need;
  }
  template <class T>
  void arg(cl_kernel k, int i, T v) {
    check(clSetKernelArg(k, i, sizeof(v), &v), "kernel argument");
  }
  // gemv4 / gemvm: one output row per work-item and ks work-items per row (local size (256 / ks up to 64) x ks)
  void dispatch_rows(cl_kernel k, int rows, int K) {
    const size_t ks = size_t(split_k(K)), l0 = std::min<size_t>(64, 256 / ks);
    size_t local[2] = {l0, ks}, global[2] = {(size_t(rows) + l0 - 1) / l0 * l0, ks};
    check(clEnqueueNDRangeKernel(queue, k, 2, nullptr, global, local, 0, nullptr, event_for(k)), "dispatch");
  }
  // K shares per row, measured on the Adreno 830: 2 for K <= 3072 (read through the image), 8 above, at least 8
  // blocks of 32 each. From K alone, so a fused call, a single projection and a verify step add in the same order;
  // XTG_GEMV_KS (1, 2, 4 or 8) overrides it
  int split_k(int K) const {
    int ks = gemv_ks ? gemv_ks : K <= 3072 ? 2 : 8;
    while (ks > 1 && K / 32 / ks < 8) ks /= 2;
    return ks;
  }
  // the decode GEMV for K: through the codes' images where every projection has one (K <= 3072)
  cl_kernel gemv_for(int K, const std::vector<const Weight*>& bs) const {
    if (K > 3072 || no_image) return gemv4;
    for (auto b : bs)
      if (!b->image) return gemv4;
    return gemv4i;
  }
  int gemv_ks = 0;
  bool no_image = false;   // XTG_GEMV_NOIMAGE: decode through global loads only
  void dispatch(cl_kernel k, int m, int n) {
    size_t local[2] = {16, 16}, global[2] = {size_t((n + 63) / 64) * 16, size_t((m + 63) / 64) * 16};
    if (m == 1) {
      local[0] = 128;
      global[0] = size_t((n + 3) / 4) * 128;
    }
    check(clEnqueueNDRangeKernel(queue, k, m == 1 ? 1 : 2, nullptr, global, local, 0, nullptr, event_for(k)), "dispatch");
  }
};

OpenClBackend::OpenClBackend() : impl_(std::make_unique<Impl>()) { impl_->init(); }
OpenClBackend::~OpenClBackend() = default;
std::string OpenClBackend::device_name() const { return impl_->name; }
size_t OpenClBackend::weight_bytes() const { return impl_->weight_size; }
void OpenClBackend::prepare(const Nvfp4Weight& w) {
  if (!w.codes) return;
  auto& p = *impl_;
  if (p.weights.count(w.codes)) return;
  XTG_CHECK(w.k > 0 && w.k % 32 == 0 && w.n_out > 0, "Invalid GPU NVFP4 shape");
  // to the k-major layout of the kernels: block b of row r (32 codes, two scales) goes to b * N + r
  const size_t n = size_t(w.n_out), blocks = size_t(w.k) / 32;
  std::vector<uint8_t> codes(n * w.k / 2), scales(n * w.k / 16);
  for (size_t r = 0; r < n; ++r)
    for (size_t b = 0; b < blocks; ++b) {
      std::memcpy(&codes[(b * n + r) * 16], w.codes + r * (w.k / 2) + b * 16, 16);
      std::memcpy(&scales[(b * n + r) * 2], w.scales + r * (w.k / 16) + b * 2, 2);
    }
  auto& b = p.weights[w.codes];
  b.codes = p.copy(codes.data(), codes.size());
  b.scales = p.copy(scales.data(), scales.size());
  if (n * blocks <= p.image_max) {   // a view, no copy
    cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc desc{};
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width = n * blocks;
    desc.buffer = b.codes;
    cl_int e;
    b.image = p.clCreateImage(p.context, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
    if (e != CL_SUCCESS) b.image = nullptr;
  }
}
void OpenClBackend::prepare(const Bf16Weight& w) {
  auto& p = *impl_;
  if (p.weights.count(w.data)) return;
  XTG_CHECK(w.k > 0 && w.k % 4 == 0 && w.n_out > 0, "Invalid GPU BF16 shape");
  auto& b = p.weights[w.data];
  b.codes = p.copy(w.data, size_t(w.n_out) * w.k * 2);
}
void OpenClBackend::matmul(const std::vector<const Nvfp4Weight*>& ws, const float* x, int m,
                           const std::vector<float*>& ys) {
  const double started = now_s();
  XTG_CHECK(!ws.empty() && ws.size() == ys.size() && m > 0, "Invalid GPU matmul arguments");
  auto& p = *impl_;
  int K = ws[0]->k;
  size_t count = 0;
  for (auto w : ws) {
    XTG_CHECK(w->k == K, "GPU input widths differ");
    prepare(*w);
    count += size_t(m) * w->n_out;
  }
  p.scratch(p.input, p.input_size, size_t(m) * K * 4);
  p.scratch(p.output, p.output_size, count * 4);
  try {
    check(p.clEnqueueWriteBuffer(p.queue, p.input, CL_FALSE, 0, size_t(m) * K * 4, x, 0, nullptr, nullptr),
          "input upload");
    int offset = 0;
    if (m == 1) {
      XTG_CHECK(ws.size() <= 4, "GPU decode supports at most four fused projections");
      std::vector<const Impl::Weight*> bs;
      for (auto w : ws) bs.push_back(&p.weights.at(w->codes));
      cl_kernel k = p.gemv_for(K, bs);
      cl_int4 counts{};
      cl_float4 scales{};
      for (int j = 0; j < 4; ++j) {
        const auto& w = *ws[j < int(ws.size()) ? j : 0];
        auto b = p.weights.at(w.codes);
        p.arg(k, j * 2, k == p.gemv4i ? b.image : b.codes);
        p.arg(k, j * 2 + 1, b.scales);
        if (j < int(ws.size())) {
          counts.s[j] = w.n_out;
          scales.s[j] = w.global;
        }
      }
      p.arg(k, 8, p.input);
      p.arg(k, 9, p.output);
      p.arg(k, 10, K);
      p.arg(k, 11, counts);
      p.arg(k, 12, scales);
      p.arg(k, 13, 0);
      p.dispatch_rows(k, int(count), K);
      for (size_t j = 0; j < ws.size(); ++j) {
        check(p.clEnqueueReadBuffer(p.queue, p.output, CL_FALSE, size_t(offset) * 4, size_t(ws[j]->n_out) * 4, ys[j], 0,
                                    nullptr, nullptr),
              "fused output read");
        offset += ws[j]->n_out;
      }
      check(p.clFinish(p.queue), "fused decode completion");
      p.decode_nv_s += now_s() - started;
      ++p.decode_nv_calls;
      return;
    }
    for (size_t j = 0; j < ws.size(); j++) {
      auto& w = *ws[j];
      auto b = p.weights.at(w.codes);
      cl_kernel k = p.gemm4;
      p.arg(k, 0, b.codes);
      p.arg(k, 1, b.scales);
      p.arg(k, 2, p.input);
      p.arg(k, 3, p.output);
      p.arg(k, 4, K);
      p.arg(k, 5, w.n_out);
      int i = 6;
      if (m != 1) p.arg(k, i++, m);
      p.arg(k, i++, w.global);
      p.arg(k, i, offset);
      p.dispatch(k, m, w.n_out);
      check(p.clEnqueueReadBuffer(p.queue, p.output, CL_FALSE, size_t(offset) * 4, size_t(m) * w.n_out * 4, ys[j], 0,
                                  nullptr, nullptr),
            "output read");
      offset += m * w.n_out;
    }
    check(p.clFinish(p.queue), "matmul completion");
  } catch (...) {
    p.clFinish(p.queue);  // asynchronous transfers must release host pointers before unwinding
    throw;
  }
}
void OpenClBackend::matmul(const Bf16Weight& w, const float* x, int m, float* y) {
  const double started = now_s();
  XTG_CHECK(m > 0, "Invalid GPU rows");
  prepare(w);
  auto& p = *impl_;
  p.scratch(p.input, p.input_size, size_t(m) * w.k * 4);
  p.scratch(p.output, p.output_size, size_t(m) * w.n_out * 4);
  try {
    check(p.clEnqueueWriteBuffer(p.queue, p.input, CL_FALSE, 0, size_t(m) * w.k * 4, x, 0, nullptr, nullptr),
          "input upload");
    cl_kernel k = m == 1 ? p.gemv16 : p.gemm16;
    p.arg(k, 0, p.weights.at(w.data).codes);
    p.arg(k, 1, p.input);
    p.arg(k, 2, p.output);
    p.arg(k, 3, w.k);
    p.arg(k, 4, w.n_out);
    int i = 5;
    if (m != 1) p.arg(k, i++, m);
    p.arg(k, i, 0);
    p.dispatch(k, m, w.n_out);
    check(p.clEnqueueReadBuffer(p.queue, p.output, CL_TRUE, 0, size_t(m) * w.n_out * 4, y, 0, nullptr, nullptr),
          "output read");
    if (m == 1) {
      p.decode_bf_s += now_s() - started;
      ++p.decode_bf_calls;
    }
  } catch (...) {
    p.clFinish(p.queue);
    throw;
  }
}

// --- device-resident path ---

OpenClBackend::Mem OpenClBackend::alloc(size_t bytes) {
  cl_int e;
  cl_mem b = impl_->clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 4), nullptr, &e);
  check(e, "device buffer");
  return b;
}
void OpenClBackend::release(Mem m) {
  if (m) impl_->clReleaseMemObject(m);
}
void OpenClBackend::write(Mem m, size_t offset, const void* src, size_t bytes) {
  if (bytes) check(impl_->clEnqueueWriteBuffer(impl_->queue, m, CL_TRUE, offset, bytes, src, 0, nullptr, nullptr), "write");
}
void OpenClBackend::read(Mem m, size_t offset, void* dst, size_t bytes) {
  if (bytes) check(impl_->clEnqueueReadBuffer(impl_->queue, m, CL_TRUE, offset, bytes, dst, 0, nullptr, nullptr), "read");
}
void OpenClBackend::copy(Mem src, size_t src_offset, Mem dst, size_t dst_offset, size_t bytes) {
  if (bytes)
    check(impl_->clEnqueueCopyBuffer(impl_->queue, src, dst, src_offset, dst_offset, bytes, 0, nullptr, nullptr), "copy");
}
void OpenClBackend::finish() {
  check(impl_->clFinish(impl_->queue), "finish");
  if (impl_->profiling) impl_->collect();
}

OpenClBackend::Kernel OpenClBackend::kernel(const char* source, const std::string& options, const char* name) {
  auto& p = *impl_;
  cl_program& prog = p.programs[{source, options}];
  if (!prog) {
    cl_int e;
    prog = p.clCreateProgramWithSource(p.context, 1, &source, nullptr, &e);
    check(e, "program");
    std::string opts = "-cl-std=CL1.2 " + options;
    if (p.clBuildProgram(prog, 1, &p.device, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
      std::string text(65536, '\0');
      p.clGetProgramBuildInfo(prog, p.device, CL_PROGRAM_BUILD_LOG, text.size() - 1, text.data(), nullptr);
      cl_program bad = prog;
      p.programs.erase({source, options});
      p.clReleaseProgram(bad);
      fail("OpenCL build of %s: %s", name, text.c_str());
    }
  }
  cl_kernel& k = p.kernels[{prog, name}];
  if (!k) {
    cl_int e;
    k = p.clCreateKernel(prog, name, &e);
    check(e, name);
  }
  return k;
}
void OpenClBackend::set_arg(Kernel k, unsigned index, size_t size, const void* value) {
  check(impl_->clSetKernelArg(k, index, size, value), "kernel argument");
}
void OpenClBackend::enqueue(Kernel k, Range global, Range local) {
  size_t g[2] = {global[0], global[1]}, l[2] = {local[0], local[1]};
  for (int i = 0; i < 2; ++i) g[i] = (g[i] + l[i] - 1) / l[i] * l[i];
  check(impl_->clEnqueueNDRangeKernel(impl_->queue, k, 2, nullptr, g, l, 0, nullptr, impl_->event_for(k)), "dispatch");
}

void OpenClBackend::matmul(const std::vector<const Nvfp4Weight*>& ws, Mem x, int m, Mem y, size_t y_offset) {
  XTG_CHECK(!ws.empty() && m > 0, "Invalid GPU matmul arguments");
  auto& p = *impl_;
  const int K = ws[0]->k;
  for (auto w : ws) {
    XTG_CHECK(w->k == K, "GPU input widths differ");
    prepare(*w);
  }
  if (m <= 16) {   // one row: gemv4 / gemv4i; up to 16 (a verify step): gemvm, the weights still read once
    XTG_CHECK(ws.size() <= 4, "GPU decode supports at most four fused projections");
    std::vector<const Impl::Weight*> bs;
    for (auto w : ws) bs.push_back(&p.weights.at(w->codes));
    cl_kernel k = m == 1 ? p.gemv_for(K, bs) : p.gemvm;
    cl_int4 counts{};
    cl_float4 scales{};
    int total = 0;
    for (int j = 0; j < 4; ++j) {
      const auto& w = *ws[j < int(ws.size()) ? j : 0];
      auto b = p.weights.at(w.codes);
      p.arg(k, j * 2, k == p.gemv4i ? b.image : b.codes);
      p.arg(k, j * 2 + 1, b.scales);
      if (j < int(ws.size())) {
        counts.s[j] = w.n_out;
        scales.s[j] = w.global;
        total += w.n_out;
      }
    }
    p.arg(k, 8, x);
    p.arg(k, 9, y);
    p.arg(k, 10, K);
    p.arg(k, 11, counts);
    p.arg(k, 12, scales);
    p.arg(k, 13, int(y_offset));
    if (m > 1) p.arg(k, 14, m);
    p.dispatch_rows(k, total, K);
    return;
  }
  size_t offset = y_offset;
  for (auto w : ws) {
    auto b = p.weights.at(w->codes);
    // the tile GEMM where the shape fits it (every Dory projection), else the 64 x 64 FP32 one
    const bool tiles = p.use_gemmt && w->n_out % 128 == 0 && K % 32 == 0;
    cl_kernel k = tiles ? p.gemmt : p.gemm4;
    p.arg(k, 0, b.codes);
    p.arg(k, 1, b.scales);
    p.arg(k, 2, x);
    p.arg(k, 3, y);
    p.arg(k, 4, K);
    p.arg(k, 5, w->n_out);
    p.arg(k, 6, m);
    p.arg(k, 7, w->global);
    p.arg(k, 8, int(offset));
    if (tiles) {
      size_t global[2] = {size_t(w->n_out / 128) * 256, size_t((m + 63) / 64)}, local[2] = {256, 1};
      check(p.clEnqueueNDRangeKernel(p.queue, k, 2, nullptr, global, local, 0, nullptr, p.event_for(k)), "dispatch");
    } else {
      p.dispatch(k, m, w->n_out);
    }
    offset += size_t(m) * size_t(w->n_out);
  }
}
void OpenClBackend::matmul(const Bf16Weight& w, Mem x, int m, Mem y) {
  prepare(w);
  auto& p = *impl_;
  if (m > 1 && m <= 16) {
    p.arg(p.gemv16m, 0, p.weights.at(w.data).codes);
    p.arg(p.gemv16m, 1, x);
    p.arg(p.gemv16m, 2, y);
    p.arg(p.gemv16m, 3, w.k);
    p.arg(p.gemv16m, 4, w.n_out);
    p.arg(p.gemv16m, 5, m);
    p.dispatch(p.gemv16m, 1, w.n_out);
    return;
  }
  cl_kernel k = m == 1 ? p.gemv16 : p.gemm16;
  p.arg(k, 0, p.weights.at(w.data).codes);
  p.arg(k, 1, x);
  p.arg(k, 2, y);
  p.arg(k, 3, w.k);
  p.arg(k, 4, w.n_out);
  int i = 5;
  if (m != 1) p.arg(k, i++, m);
  p.arg(k, i, 0);
  p.dispatch(k, m, w.n_out);
}
}  // namespace xtg
