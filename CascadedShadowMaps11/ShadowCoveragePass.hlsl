#include <RenderCascadeScene.hlsl>

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static const uint COMPUTE_NUM_THREAD_X = 16;
static const uint COMPUTE_NUM_THREAD_Y = 16;

static const float2 COVERAGE_MAP_SIZE = float2( 128, 128 );

groupshared uint g_sharedData[COMPUTE_NUM_THREAD_X][COMPUTE_NUM_THREAD_Y];

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<uint>    g_txCoverageMap              : register(u0);
Texture2D<float>	 g_txDepthMap		          : register(t0);

SamplerState DepthSampler : register(s0);

bool compareHash( uint4 a, uint4 b )
{
	bool result = true;

	[unroll]
	// Check speed is more important than check accuracy
	for (int i = 0; i < 4; i++)
		result = result && (a[i] == b[0]);
	
	return result;
}

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1)]
void main( uint3 dispatchThreadId : SV_DispatchThreadID, uint3 threadID : SV_GroupThreadId, uint3 gID : SV_GroupID )
{ 
	int2 outputIndex[4];
	uint outputData[4] = { 0, 0, 0, 0 };
	uint dataHash[4] = { 0, 0, 0, 0 };
	uint4 dataHashVector;

	const uint2 dispatchSampleIndex = dispatchThreadId.xy * 2;

	const float2 screenUV = dispatchSampleIndex * m_ScreenSize.zw;
	const float4 zwDepthGather4 = g_txDepthMap.Gather( DepthSampler, screenUV );
	
	// Little bit coarse values (because screenUV should be calculated for each sample)
	const float3 viewRayLeft = lerp( m_CameraDirs[0], m_CameraDirs[1], screenUV.y ).xyz;
	const float3 viewRayRight = lerp( m_CameraDirs[2], m_CameraDirs[3], screenUV.y ).xyz;
	const float3 viewRay = lerp( viewRayLeft, viewRayRight, screenUV.x );

	[unroll]
	for (uint sampleId = 0; sampleId < 4; sampleId++)
	{
		static const float EPS = 0.000001;
		const float zwDepth = zwDepthGather4[sampleId];

		[flatten]
		if (zwDepth > 1.0 - EPS)
			continue;
	
		const float linearDepth = m_mProj[3][2] / (zwDepth - m_mProj[2][2]);

		const float4 worldPos = float4(m_CameraPosition.xyz + viewRay * linearDepth, 1.0);
		const float2 vShadowTexCoordViewSpace = mul( worldPos, m_mShadow ).xy;

		[unroll]
		for (int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex)
		{
			float2 vShadowTexCoord01 = vShadowTexCoordViewSpace * m_vCascadeScale[iCascadeIndex].xy;
			vShadowTexCoord01 += m_vCascadeOffset[iCascadeIndex].xy;
			
			float2 vShadowTexCoord = vShadowTexCoord01;
			vShadowTexCoord.x = (vShadowTexCoord.x + (float)iCascadeIndex) * m_fShadowPartitionSize;// precomputed (float)iCascadeIndex / (float)CASCADE_CNT
			
			[branch]
			if ( min( vShadowTexCoord.x, vShadowTexCoord.y ) > m_fMinBorderPadding &&
				 max( vShadowTexCoord.x, vShadowTexCoord.y ) < m_fMaxBorderPadding )
			{
				outputIndex[sampleId] = int2( vShadowTexCoord01 * COVERAGE_MAP_SIZE );
				outputData[sampleId] = 1 << iCascadeIndex;
				dataHash[sampleId] = (outputIndex[sampleId].x << 16) + (outputIndex[sampleId].y << 8) + outputData[sampleId];
				break;
			}
		}
	}	
	dataHashVector = uint4(dataHash[0], dataHash[1], dataHash[2], dataHash[3]);

	// All samples are discarded
	if( all( dataHashVector == uint4( 0, 0, 0, 0 ) ) )
		return;
	
	g_sharedData[threadID.x][threadID.y] = dataHash[0];
	
	GroupMemoryBarrier();
	
	[branch]
	if (threadID.x != COMPUTE_NUM_THREAD_X - 1)
	{
		uint sdata = g_sharedData[threadID.x+1][threadID.y];
		[branch]
		if ( all( dataHashVector == sdata ) )
			return;
	}

	[branch]
	if(threadID.y != COMPUTE_NUM_THREAD_Y - 1)
	{
		[branch]
		if ( all( dataHashVector == g_sharedData[threadID.x][threadID.y + 1] ) )
			return;
	}

	static bool debugOutput = false;
	static uint2 sampleOffsets[4] = {uint2(0, 1), uint2(1, 1), uint2(1, 0), uint2(0, 0)};

	{
		[unroll]
		for(uint sampleId = 0; sampleId < 4; sampleId++)
		{
			const uint2 outputSampleIndex = debugOutput ? dispatchSampleIndex + sampleOffsets[sampleId] : outputIndex[sampleId];
			const uint2 outpuData = debugOutput ? dataHash[sampleId] : outputData[sampleId];

			[branch]
			if(sampleId == 0 || dataHash[sampleId] != dataHash[0])
				InterlockedOr( g_txCoverageMap[dispatchSampleIndex + sampleOffsets[sampleId]], dataHash[sampleId] );
		}
	}
}