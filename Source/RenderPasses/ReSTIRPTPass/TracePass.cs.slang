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

ParameterBlock<PathTracer> gPathTracer;
RWBuffer<uint> gPathIDs;

cbuffer CB
{
    int gSampleId;
}

void tracePath(uint pathID, inout PathReservoir giReservoir, uint sampleIdx)
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
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    // if (all(pixel >= (gPathTracer.params.frameDim))) return;

    uint2 frameDim = gPathTracer.params.frameDim;
    uint pathIDIndex = 1 + pixel.x + (pixel.y * frameDim.x);

    // if (pathIDIndex % 2 == 0) return;

    if (pathIDIndex > gPathIDs[0]) return;
    uint pathID;
    pathID = gPathIDs[pathIDIndex];
    pixel.x = pathID & 0xfff;
    pixel.y = pathID >> 12;
    // pathID = (pixel.y << 12) | pixel.x;

    // Skip pixel if there is no hit in the vbuffer. // Should be skippable now
    HitInfo hit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!hit.isValid()) return;

    PathReservoir giReservoir;

    static const uint itersPerShaderPass = PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing ? kSamplesPerPixel : kCandidateSamples;

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
