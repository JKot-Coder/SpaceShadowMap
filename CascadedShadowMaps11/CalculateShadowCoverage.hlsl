#include <RenderCascadeScene.hlsl>

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static const uint COMPUTE_NUM_THREAD_X = 8;
static const uint COMPUTE_NUM_THREAD_Y = 8;

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<uint>    g_txCoverageMap              : register(u0);
Texture2D<float>	 g_txDepthMap		          : register(t0);

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1)]
void main( uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupID : SV_GroupID )
{ 
	const float zwDepth = g_txDepthMap.Load( int3(dispatchThreadId.xy, 0) );
	const float linearDepth = m_mProj[3][2] / (zwDepth - m_mProj[2][2]);

	const float2 screenUV = dispatchThreadId.xy * m_ScreenSize.zw;

	const float3 viewRayLeft = lerp( m_CameraDirs[0], m_CameraDirs[1], screenUV.y ).xyz;
	const float3 viewRayRight = lerp( m_CameraDirs[2], m_CameraDirs[3], screenUV.y ).xyz;
	const float3 viewRay = lerp( viewRayLeft, viewRayRight, screenUV.x );

	const float4 worldPos = float4(m_CameraPosition.xyz + viewRay * linearDepth, 1.0);
	const float4 vShadowTexCoordViewSpace = mul( worldPos, m_mShadow );
	
	const float2 shadowTexCoordLastCascade = vShadowTexCoordViewSpace * m_vCascadeScale[CASCADE_COUNT_FLAG - 1] + m_vCascadeOffset[CASCADE_COUNT_FLAG - 1];
	const int2 outputIndex = int2(shadowTexCoordLastCascade / m_fTexelSize / float2(COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y));

	[unroll]
	for (int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex)
	{
		float4 vShadowTexCoord = vShadowTexCoordViewSpace * m_vCascadeScale[iCascadeIndex];
		vShadowTexCoord += m_vCascadeOffset[iCascadeIndex];

		vShadowTexCoord.x *= m_fShadowPartitionSize;  // precomputed (float)iCascadeIndex / (float)CASCADE_CNT
		vShadowTexCoord.x += (m_fShadowPartitionSize * (float)iCascadeIndex);

		if (min(vShadowTexCoord.x, vShadowTexCoord.y) > m_fMinBorderPadding
			&& max(vShadowTexCoord.x, vShadowTexCoord.y) < m_fMaxBorderPadding)
		{
			InterlockedOr( g_txCoverageMap[outputIndex.xy], 1 << iCascadeIndex );
			return;
		}
	}
}