/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Scene.Raytracing;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import PathTracer;
import PathState;
import Params;
import PathReservoir;
import Utils.Sampling.TinyUniformSampleGenerator;

ParameterBlock<PathTracer> gPathTracer;

// Adaptive ReSTIR
Buffer<uint> gRISPathIDs;
RWTexture2D<float4> outputColor;                    ///< Output color buffer if kSamplesPerPixel == 1.
RWStructuredBuffer<PathReservoir> outputReservoirs; // reservoir for next pass

cbuffer CB
{
    int gSampleId;
    int gMergePaths;
}

PathReservoir tracePath(uint pathID, inout PathReservoir giReservoir, uint sampleIdx)
{
    PathState path;
    gPathTracer.generatePath(pathID, path);

    printSetPixel(path.getPixel());
    logSetPixel(path.getPixel());

    while (path.isActive())
    {
        if (path.isHit())
        {
            gPathTracer.handleHit(path);
            if (path.isTerminated()) break;
            gPathTracer.nextVertex(path);
        }
        else
        {
            gPathTracer.handleMiss(path);
        }
    }

    gPathTracer.finalize(path);
    gPathTracer.writeOutput(path, giReservoir, sampleIdx);

    return giReservoir;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;

    uint2 frameDim = gPathTracer.params.frameDim;
    uint pathIDIndex = 1 + pixel.x + (pixel.y * frameDim.x);
    if (pathIDIndex > gRISPathIDs[0]) return;

    uint pathID = gRISPathIDs[pathIDIndex];
    uint sampleCount = pathID >> 24;
    pathID &= 0x00FFFFFF;
    pixel.x = pathID & 0x00000FFF;
    pixel.y = pathID >> 12;

    // Skip pixel if there is no hit in the vbuffer. // Should be skippable now
    HitInfo hit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!hit.isValid()) return;

    PathReservoir giReservoir;

    uint itersPerShaderPass = PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing ? kSamplesPerPixel : max(1U, sampleCount);

    if (itersPerShaderPass == 1)
    {
        // Handle fixed 1 spp case.
        pathID = pathID | (gSampleId << 24);
        tracePath(pathID, giReservoir, 0);
    }
    else//(itersPerShaderPass > 1)
    {
        // Handle fixed multiple spp case.
        for (uint sampleIdx = 0; sampleIdx < itersPerShaderPass; ++sampleIdx)
        {
            pathID = pathID | (sampleIdx << 24);
            tracePath(pathID, giReservoir, sampleIdx);
        }
    }
}
