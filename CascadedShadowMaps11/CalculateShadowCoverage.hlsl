//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<uint>    g_txCoverageMap              : register(u0);
Texture2D			 g_txDepthMap		          : register(t0);

static const int COMPUTE_NUM_THREAD_X = 8;
static const int COMPUTE_NUM_THREAD_Y = 8;

SamplerState PointSampler {
	Filter = MIN_MAG_MIP_POINT;
	AddressU = Clamp;
	AddressV = Clamp;
};

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1)]
void main( uint3 index : SV_DispatchThreadID )
{
	g_txCoverageMap[index.xy] = index.x;
}