#include <RenderCascadeScene.hlsl>

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static const uint COMPUTE_NUM_THREAD_X = 16;
static const uint COMPUTE_NUM_THREAD_Y = 16;

static const float2 COVERAGE_MAP_SIZE = float2( 128, 128 );

groupshared uint4 g_sharedData[COMPUTE_NUM_THREAD_X+1][COMPUTE_NUM_THREAD_Y]; // Plus one for minimize bank conflicts

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<uint>    g_txCoverageMap              : register(u0);
Texture2D<float>	 g_txDepthMap		          : register(t0);

SamplerState DepthSampler
{
	Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	AddressU = Clamp;
	AddressV = Clamp;
};

bool compareHash( uint a[4], uint4 b )
{
	bool result = 0;

	[unroll]
	for (int i = 0; i < 4; i++)
		result = a[i] == b[0] | a[i] == b[1] | a[i] == b[2] | a[i] == b[3];
	
	return result;
}

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1)]
void main( uint3 dispatchThreadId : SV_DispatchThreadID, uint3 threadID : SV_GroupThreadId, uint3 gID : SV_GroupID )
{ 
	
	
	int2 outputIndex[4];
	uint outputData[4] = { 0, 0, 0, 0 };
	uint dataHash[4] = { 0, 0, 0, 0 };

	const uint2 dispatchSampleIndex = dispatchThreadId.xy * 2;

	const float2 screenUV = dispatchSampleIndex * m_ScreenSize.zw;
	const float4 zwDepthGather4 = g_txDepthMap.Gather( DepthSampler, screenUV );
	
	const float3 viewRayLeft = lerp( m_CameraDirs[0], m_CameraDirs[1], screenUV.y ).xyz;
	const float3 viewRayRight = lerp( m_CameraDirs[2], m_CameraDirs[3], screenUV.y ).xyz;
	const float3 viewRay = lerp( viewRayLeft, viewRayRight, screenUV.x );
	
	[unroll]
	for (uint sampleId = 0; sampleId < 4; sampleId++)
	{
		static const float EPS = 0.000001;
		const float zwDepth = zwDepthGather4[sampleId];

		if (zwDepth > 1.0 - EPS)
			continue;

		const float linearDepth = m_mProj[3][2] / (zwDepth - m_mProj[2][2]);

		const float4 worldPos = float4(m_CameraPosition.xyz + viewRay * linearDepth, 1.0);
		const float4 vShadowTexCoordViewSpace = mul( worldPos, m_mShadow );
		const float2 shadowTexCoordLastCascade = vShadowTexCoordViewSpace * m_vCascadeScale[CASCADE_COUNT_FLAG - 1] + m_vCascadeOffset[CASCADE_COUNT_FLAG - 1];

		[unroll]
		for (int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex)
		{
			float4 vShadowTexCoord = vShadowTexCoordViewSpace * m_vCascadeScale[iCascadeIndex];
			vShadowTexCoord += m_vCascadeOffset[iCascadeIndex];

			vShadowTexCoord.x *= m_fShadowPartitionSize;  // precomputed (float)iCascadeIndex / (float)CASCADE_CNT
			vShadowTexCoord.x += (m_fShadowPartitionSize * (float)iCascadeIndex);

			if ( min( vShadowTexCoord.x, vShadowTexCoord.y ) > m_fMinBorderPadding &&
				 max( vShadowTexCoord.x, vShadowTexCoord.y ) < m_fMaxBorderPadding )
			{
				outputIndex[sampleId] = int2( shadowTexCoordLastCascade * COVERAGE_MAP_SIZE );
				outputData[sampleId] = 1 << iCascadeIndex;
				dataHash[sampleId] = (outputIndex[sampleId].x << 16) + (outputIndex[sampleId].y << 8) + outputData[sampleId];
				break;
			}
		}
	}

	g_sharedData[threadID.x][threadID.y] = uint4(dataHash[0], dataHash[1], dataHash[2], dataHash[3]);
	
	if ( uint(dataHash[0] | dataHash[1] | dataHash[2] | dataHash[3]) == 0 )
		return;

	GroupMemoryBarrier();
	
	if (threadID.x != COMPUTE_NUM_THREAD_X - 1 || threadID.y != COMPUTE_NUM_THREAD_X - 1)
	{
		uint4 sdata = g_sharedData[0][threadID.y];
		if ( threadID.x != 0 && compareHash( dataHash, sdata ) )
			return;

		sdata = g_sharedData[threadID.x][0];
		if ( threadID.y != 0 && compareHash( dataHash, sdata ) )
			return;

		sdata = g_sharedData[COMPUTE_NUM_THREAD_X - 1][threadID.y];
		if ( compareHash( dataHash, sdata ) )
			return;

		sdata = g_sharedData[threadID.x][COMPUTE_NUM_THREAD_Y - 1];
		if ( compareHash( dataHash, sdata ) )
			return;
	}

	static uint2 sampleOffsets[4] = { uint2(0, 1), uint2(1, 1), uint2(1, 0), uint2(0, 0) };

	[unroll]
	for (int i = 0; i < 4; i++)
		InterlockedOr( g_txCoverageMap[outputIndex[i].xy + sampleOffsets[i]], outputData[i] );
}