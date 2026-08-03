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
RWBuffer<uint> gPathIDs;
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
    if (pathIDIndex > gPathIDs[0]) return;

    uint pathID = gPathIDs[pathIDIndex];
    pixel.x = pathID & 0xfff;
    pixel.y = pathID >> 12;

    // Skip pixel if there is no hit in the vbuffer. // Should be skippable now
    HitInfo hit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!hit.isValid()) return;

    var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1) * gPathTracer.params.seed + kCandidateSamples);
    const uint reservoirOffset = gPathTracer.params.getReservoirOffset(pixel);
    PathReservoir temporalReservoir = outputReservoirs[reservoirOffset];
    PathReservoir RISReservoir;
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
    
    // Debug
    // if (RISReservoir.weight < 0.f)
    // {
    //     outputColor[pixel] = float4(float3(0.8f, 0.2f, 0.8f), 1.0f);
    // }

    // if (temporalReservoir.weight < 0.f)
    // {
    //     outputColor[pixel] = float4(float3(0.0f, 0.8f, 0.0f), 1.0f);
    // }

    if (temporalReservoir.weight > 0.f && gMergePaths == 1)
    {
        PathReservoir RISReservoir = outputReservoirs[reservoirOffset];
        temporalReservoir.M = min(20.0f * RISReservoir.M, temporalReservoir.M);
        temporalReservoir.prepareMerging();
        temporalReservoir.mergeInSamplePixel(RISReservoir, sg);
        temporalReservoir.finalizeRIS();

        outputColor[pixel] = float4((temporalReservoir.F * temporalReservoir.weight) / kCandidateSamples, 1.0f);
        outputReservoirs[reservoirOffset] = temporalReservoir;

        //Debug
        // outputColor[pixel] = float4(float3(0.0f), 1.0f);
    }
    else if (temporalReservoir.weight <= 0.f)
    {
        // outputColor[pixel] = float4(float3(0.0f, 0.8f, 0.8f), 1.0f);
    }
}
