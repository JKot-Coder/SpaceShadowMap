// Copyright (c) 2011 by Nathan Reed.
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//	this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "DXUT.h"

#include <chrono>
#include "gpuprofiler.hpp"

#define ErrorPrintf printf
#define DebugPrintf printf

float Time()	// Retrieve time in seconds, using QueryPerformanceCounter or whatever
{
	auto now = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() * 1e-6f;
}

CGpuProfiler::CGpuProfiler ()
:	m_iFrameQuery(0),
	m_iFrameCollect(-1),
	m_frameCountAvg(0),
	m_tBeginAvg(0.0f)
{
	memset(m_apQueryTsDisjoint, 0, sizeof(m_apQueryTsDisjoint));
	memset(m_apQueryTs, 0, sizeof(m_apQueryTs));
	memset(m_adT, 0, sizeof(m_adT));
	memset(m_adTAvg, 0, sizeof(m_adT));
	memset(m_adTTotalAvg, 0, sizeof(m_adT));
}

bool CGpuProfiler::Init ( ID3D11Device* pd3dDevice )
{
	// Create all the queries we'll need

	D3D11_QUERY_DESC queryDesc = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };

	if (FAILED(pd3dDevice->CreateQuery(&queryDesc, &m_apQueryTsDisjoint[0])))
	{
		ErrorPrintf("Could not create timestamp disjoint query for frame 0!");
		return false;
	}

	if (FAILED(pd3dDevice->CreateQuery(&queryDesc, &m_apQueryTsDisjoint[1])))
	{
		ErrorPrintf("Could not create timestamp disjoint query for frame 1!");
		return false;
	}

	queryDesc.Query = D3D11_QUERY_TIMESTAMP;

	for (GTS gts = GTS_BeginFrame; gts < GTS_Max; gts = GTS(gts + 1))
	{
		if (FAILED(pd3dDevice->CreateQuery(&queryDesc, &m_apQueryTs[gts][0])))
		{
			ErrorPrintf("Could not create start-frame timestamp query for GTS %d, frame 0!", gts);
			return false;
		}

		if (FAILED(pd3dDevice->CreateQuery(&queryDesc, &m_apQueryTs[gts][1])))
		{
			ErrorPrintf("Could not create start-frame timestamp query for GTS %d, frame 1!", gts);
			return false;
		}
	}

	return true;
}

void CGpuProfiler::Shutdown ()
{
	if (m_apQueryTsDisjoint[0])
		m_apQueryTsDisjoint[0]->Release();

	if (m_apQueryTsDisjoint[1])
		m_apQueryTsDisjoint[1]->Release();

	for (GTS gts = GTS_BeginFrame; gts < GTS_Max; gts = GTS(gts + 1))
	{
		if (m_apQueryTs[gts][0])
			m_apQueryTs[gts][0]->Release();

		if (m_apQueryTs[gts][1])
			m_apQueryTs[gts][1]->Release();
	}
}

void CGpuProfiler::BeginFrame ( ID3D11DeviceContext* pContext )
{
	pContext->Begin(m_apQueryTsDisjoint[m_iFrameQuery]);
	Timestamp( pContext, GTS_BeginFrame);
}

void CGpuProfiler::Timestamp ( ID3D11DeviceContext* pContext, GTS gts)
{
	pContext->End(m_apQueryTs[gts][m_iFrameQuery]);
}

void CGpuProfiler::EndFrame ( ID3D11DeviceContext* pContext )
{
	Timestamp( pContext, GTS_EndFrame);
	pContext->End(m_apQueryTsDisjoint[m_iFrameQuery]);

	++m_iFrameQuery &= 1;
}

void CGpuProfiler::WaitForDataAndUpdate ( ID3D11DeviceContext* pContext )
{
	if (m_iFrameCollect < 0)
	{
		// Haven't run enough frames yet to have data
		m_iFrameCollect = 0;
		return;
	}

	// Wait for data
	while (pContext->GetData(m_apQueryTsDisjoint[m_iFrameCollect], NULL, 0, 0) == S_FALSE)
	{
		Sleep(1);
	}

	int iFrame = m_iFrameCollect;
	++m_iFrameCollect &= 1;

	D3D11_QUERY_DATA_TIMESTAMP_DISJOINT timestampDisjoint;
	if (pContext->GetData(m_apQueryTsDisjoint[iFrame], &timestampDisjoint, sizeof(timestampDisjoint), 0) != S_OK)
	{
		DebugPrintf("Couldn't retrieve timestamp disjoint query data");
		return;
	}

	if (timestampDisjoint.Disjoint)
	{
		// Throw out this frame's data
		DebugPrintf("Timestamps disjoint");
		return;
	}

	UINT64 timestampPrev;
	if (pContext->GetData(m_apQueryTs[GTS_BeginFrame][iFrame], &timestampPrev, sizeof(UINT64), 0) != S_OK)
	{
		DebugPrintf("Couldn't retrieve timestamp query data for GTS %d", GTS_BeginFrame);
		return;
	}

	for (GTS gts = GTS(GTS_BeginFrame + 1); gts < GTS_Max; gts = GTS(gts + 1))
	{
		UINT64 timestamp;
		if (pContext->GetData(m_apQueryTs[gts][iFrame], &timestamp, sizeof(UINT64), 0) != S_OK)
		{
			DebugPrintf("Couldn't retrieve timestamp query data for GTS %d", gts);
			return;
		}

		m_adT[gts] = float(timestamp - timestampPrev) / float(timestampDisjoint.Frequency);
		timestampPrev = timestamp;

		m_adTTotalAvg[gts] += m_adT[gts];
	}

	++m_frameCountAvg;
	if (Time() > m_tBeginAvg + 0.5f)
	{
		for (GTS gts = GTS_BeginFrame; gts < GTS_Max; gts = GTS(gts + 1))
		{
			m_adTAvg[gts] = m_adTTotalAvg[gts] / m_frameCountAvg;
			m_adTTotalAvg[gts] = 0.0f;
		}

		m_frameCountAvg = 0;
		m_tBeginAvg = Time();
	}
}