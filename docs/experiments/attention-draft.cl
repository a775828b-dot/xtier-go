// UNVALIDATED DRAFT. Not built, not wired into the runtime.
// Requires fp8() from engine/src/opencl.cpp plus an unimplemented host-side KV mirror.
// No GPU attention speed or correctness claim is made for this file.
float cache_value(__global const uchar*p,__global const float*s,int slot,int h,int d,int H,int D,int fp8kv){
    size_t index=((size_t)slot*H+h)*D+d;
    return fp8kv?fp8(p[index])*s[(size_t)slot*H+h]:as_float((uint)((__global const ushort*)p)[index]<<16);
}
// Online softmax, 128 keys per tile. QK and PV both accumulate in FP32.
__kernel void attend(__global const float*q,__global const float*k,__global const float*v,
                     __global const uchar*ck,__global const uchar*cv,
                     __global const float*ks,__global const float*vs,__global float*out,
                     int heads,int H,int D,int pos,int window,int slots,int fp8kv,float scale){
    int lid=get_local_id(0),r=get_group_id(0),head=get_group_id(1),h=head/(heads/H),t=pos+r;
    int lo=window>0?max(0,t-window+1):0;
    __local float red[128],prob[128];
    float maximum=-INFINITY,denom=0;float2 acc=0;
    int d=lid*2;
    for(int base=lo;base<=t;base+=128){
        int key=base+lid;float dot=0;
        if(key<=t){
            float4 sum=0;
            for(int z=0;z<D;z+=4){
                float4 kk;
                if(key>=pos)kk=vload4(0,k+((size_t)(key-pos)*H+h)*D+z);
                else {int slot=key%slots;size_t idx=((size_t)slot*H+h)*D+z;
                    if(fp8kv){uchar4 b=vload4(0,ck+idx);kk=(float4)(fp8(b.x),fp8(b.y),fp8(b.z),fp8(b.w))*ks[(size_t)slot*H+h];}
                    else kk=as_float4(convert_uint4(vload4(0,(__global const ushort*)ck+idx))<<16);}
                sum=mad(kk,vload4(0,q+((size_t)r*heads+head)*D+z),sum);
            }
            dot=(sum.x+sum.y+sum.z+sum.w)*scale;
        }else dot=-INFINITY;
        red[lid]=dot;barrier(CLK_LOCAL_MEM_FENCE);
        for(int step=64;step;step/=2){if(lid<step)red[lid]=fmax(red[lid],red[lid+step]);barrier(CLK_LOCAL_MEM_FENCE);}
        float next=fmax(maximum,red[0]),corr=isinf(maximum)?0:exp(maximum-next);
        float p=key<=t?exp(dot-next):0;prob[lid]=p;
        barrier(CLK_LOCAL_MEM_FENCE);
        red[lid]=p;barrier(CLK_LOCAL_MEM_FENCE);
        for(int step=64;step;step/=2){if(lid<step)red[lid]+=red[lid+step];barrier(CLK_LOCAL_MEM_FENCE);}
        denom=denom*corr+red[0];acc*=corr;maximum=next;
        if(d<D)for(int j=0;j<min(128,t-base+1);j++){
            int keyj=base+j;float2 vv;
            if(keyj>=pos)vv=vload2(0,v+((size_t)(keyj-pos)*H+h)*D+d);
            else {int slot=keyj%slots;vv=(float2)(cache_value(cv,vs,slot,h,d,H,D,fp8kv),cache_value(cv,vs,slot,h,d+1,H,D,fp8kv));}
            acc=mad(vv,(float2)(prob[j]),acc);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(d<D)vstore2(acc/denom,0,out+((size_t)r*heads+head)*D+d);
}
