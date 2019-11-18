//--------------------------------------------------------------------------------------
// Input / Output structures
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
	float2 vPosition    : POSITION;
	float2 vTexcoord    : TEXCOORD0;
};

struct VS_OUTPUT
{
	float4 vPosition    : SV_POSITION;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
VS_OUTPUT VSMain( VS_INPUT Input )
{
	VS_OUTPUT Output;

	// There is nothing special here, just transform and write out the depth.
	Output.vPosition = float4( Input.vPosition, 0, 1 );

	return Output;
}