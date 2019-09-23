#include <RenderCascadeScene.hlsl>

//--------------------------------------------------------------------------------------
// Globals
//--------------------------------------------------------------------------------------
static const uint COMPUTE_NUM_THREAD_X = 8;
static const uint COMPUTE_NUM_THREAD_Y = 8;

//--------------------------------------------------------------------------------------
// Textures and Samplers
//--------------------------------------------------------------------------------------
RWTexture2D<float2>    g_txSSUVMap	: register(u0);
Texture2D<float>	 g_txDepthMap	: register(t0);

SamplerState DepthSampler : register(s0);

[numthreads( COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, 1 )]
void main( uint3 dispatchThreadId : SV_DispatchThreadID, uint3 threadID : SV_GroupThreadId, uint3 gID : SV_GroupID )
{
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
		static uint2 sampleOffsets[4] = { uint2(0, 1), uint2(1, 1), uint2(1, 0), uint2(0, 0) };
		const uint2 outputIndex = dispatchSampleIndex + sampleOffsets[sampleId];

		static const float EPS = 0.000001;
		const float zwDepth = zwDepthGather4[sampleId];
		
		float2 shadowTexCoord = float2(-1, -1);

		[branch]
		if (zwDepth > 1.0 - EPS)
		{
			g_txSSUVMap[outputIndex] = shadowTexCoord;
			continue;
		}

		const float linearDepth = m_mProj[3][2] / (zwDepth - m_mProj[2][2]);
		const float4 worldPos = float4(m_CameraPosition.xyz + viewRay * linearDepth, 1.0);
		const float2 vShadowTexCoordViewSpace = mul( worldPos, m_mShadow ).xy;

		[unroll]
		for (int iCascadeIndex = 0; iCascadeIndex < CASCADE_COUNT_FLAG; ++iCascadeIndex)
		{
			float2 vShadowTexCoord01 = vShadowTexCoordViewSpace * m_vCascadeScale[iCascadeIndex].xy;
			vShadowTexCoord01 += m_vCascadeOffset[iCascadeIndex].xy;

			float2 vShadowTexCoord = vShadowTexCoord01;
			vShadowTexCoord.x *= m_fShadowPartitionSize;
			vShadowTexCoord.x = (vShadowTexCoord.x + (float)iCascadeIndex) * m_fShadowPartitionSize;// precomputed (float)iCascadeIndex / (float)CASCADE_CNT

			[branch]
			if (min( vShadowTexCoord.x, vShadowTexCoord.y ) > m_fMinBorderPadding &&
				max( vShadowTexCoord.x, vShadowTexCoord.y ) < m_fMaxBorderPadding)
			{
				shadowTexCoord = vShadowTexCoord;
				break;
			}
		}

		g_txSSUVMap[outputIndex] = shadowTexCoord;
	}
}