#include <RenderCascadeScene.hlsl>

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static const uint COMPUTE_NUM_THREAD_X = 16;
static const uint COMPUTE_NUM_THREAD_Y = 16;

static const float2 COVERAGE_MAP_SIZE = float2( 128, 128 );

groupshared uint g_sharedData[COMPUTE_NUM_THREAD_X+1][COMPUTE_NUM_THREAD_Y]; // Plus one for minimize bank conflicts

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<uint>    g_txCoverageMap              : register(u0);
Texture2D<float>	 g_txDepthMap		          : register(t0);

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1)]
void main( uint3 dispatchThreadId : SV_DispatchThreadID, uint3 threadID : SV_GroupThreadId, uint3 gID : SV_GroupID )
{ 
	int2 outputIndex = 0;
	uint outputData = 0;
	uint dataHash = 0;

	const float EPS = 0.000001;
	const float zwDepth = g_txDepthMap.Load( int3(dispatchThreadId.xy, 0) );

	if (zwDepth < 1.0 - EPS)
	{
		const float linearDepth = m_mProj[3][2] / ( zwDepth - m_mProj[2][2] );
		const float2 screenUV = dispatchThreadId.xy * m_ScreenSize.zw;

		const float3 viewRayLeft = lerp( m_CameraDirs[0], m_CameraDirs[1], screenUV.y ).xyz;
		const float3 viewRayRight = lerp( m_CameraDirs[2], m_CameraDirs[3], screenUV.y ).xyz;
		const float3 viewRay = lerp( viewRayLeft, viewRayRight, screenUV.x );

		const float4 worldPos = float4( m_CameraPosition.xyz + viewRay * linearDepth, 1.0 );
		const float4 vShadowTexCoordViewSpace = mul( worldPos, m_mShadow );

		const float2 shadowTexCoordLastCascade = vShadowTexCoordViewSpace * m_vCascadeScale[CASCADE_COUNT_FLAG - 1] + m_vCascadeOffset[CASCADE_COUNT_FLAG - 1];
		
		[unroll]
		for (int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex)
		{
			float4 vShadowTexCoord = vShadowTexCoordViewSpace * m_vCascadeScale[iCascadeIndex];
			vShadowTexCoord += m_vCascadeOffset[iCascadeIndex];

			vShadowTexCoord.x *= m_fShadowPartitionSize;  // precomputed (float)iCascadeIndex / (float)CASCADE_CNT
			vShadowTexCoord.x += ( m_fShadowPartitionSize * (float)iCascadeIndex );

			if ( min( vShadowTexCoord.x, vShadowTexCoord.y ) > m_fMinBorderPadding
				&& max( vShadowTexCoord.x, vShadowTexCoord.y ) < m_fMaxBorderPadding )
			{
				outputIndex = int2( shadowTexCoordLastCascade * COVERAGE_MAP_SIZE );
				outputData = 1 << iCascadeIndex;
				dataHash = outputIndex.x << 16 + outputIndex.y << 8 + outputData;
				break;
			}
		}
	}
	g_sharedData[threadID.x][threadID.y] = dataHash;

	if (outputData == 0)
		return;

	GroupMemoryBarrier();
	
	if (threadID.x != COMPUTE_NUM_THREAD_X - 1 || threadID.y != COMPUTE_NUM_THREAD_X - 1)
	{
		if ( threadID.x != 0 && dataHash == g_sharedData[0][threadID.y] )
			return;

		if ( threadID.y != 0 && dataHash == g_sharedData[threadID.x][0] )
			return;

		if ( dataHash == g_sharedData[COMPUTE_NUM_THREAD_X - 1][threadID.y] )
			return;

		if ( dataHash == g_sharedData[threadID.x][COMPUTE_NUM_THREAD_Y - 1] )
			return;
	}

	InterlockedOr( g_txCoverageMap[outputIndex.xy], outputData );
}