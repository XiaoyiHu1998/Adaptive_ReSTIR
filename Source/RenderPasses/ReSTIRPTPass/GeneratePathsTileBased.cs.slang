/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Scene.Scene;
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.ClothMaterial;
import Rendering.Materials.HairMaterial;
import LoadShadingData;
import PathTracer;
import PathState;
import Params;
import PathReservoir;

// Shared memory variables.
static const uint kWarpCount = (kScreenTileDim.x * kScreenTileDim.y) / 32;

struct PathGenerator
{
    // Resources
    RestirPathTracerParams params; ///< Runtime parameters.
    Texture2D<PackedHitInfo> vbuffer; ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture2D<float2> motionVectors;
    RWTexture2D<float4> outputColor; ///< Output color buffer if kSamplesPerPixel == 1.

    RWStructuredBuffer<PathReservoir> outputReservoirs;   // reservoir for next pass
    RWStructuredBuffer<PathReservoir> temporalReservoirs; // reservoir from previous frame

    RWBuffer<uint> gRISPathIDs;
    RWBuffer<uint> gNonRISPathIDs;

    int gSampleId;
    uint gPatternShift;
    bool gEnableTemporalReprojection;
    float gAdaptiveTemporalHistoryCap;

    uint gPatternOneSixteenth;
    uint gPatternOneEight;
    uint gPatternQuarter;
    uint gPatternHalf;
    uint gPatternThreeQuarters;

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    // static bool kUseEnvBackground = USE_ENV_BACKGROUND;
    bool kUseEnvBackground;

    bool isValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    ShadingData getPixelShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    ShadingData getPixelTemporalShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinholePrevFrame(pixel, params.frameDim);
        PrimaryHitPacked = temporalVbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    float easingFunction(float x)
    {
        return x * x * x * x * x;
    }

    // Entrypoint
    void execute(const uint2 startingPixel)
    {
        // Determine target RIS candidate count using data from all pixels in tile
        float targetSampleCount = 0.0f;
        for (int i = 0; i < kAdaptiveTileSize; i++)
        {
            for (int j = 0; j < kAdaptiveTileSize; j++)
            {
                const uint2 pixel = startingPixel + uint2(i, j);

                Ray cameraRay;
                if (all(pixel < params.frameDim))
                {
                    cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);

                    // Load the primary hit from the V-buffer.
                    const HitInfo hit = HitInfo(vbuffer[pixel]);
                    bool hitSurface = hit.isValid();

                    if (!hitSurface && gSampleId == 0)
                    {
                        // Write background pixels.
                        writeBackground(pixel, cameraRay.dir);
                        continue;
                    }

                    // Determine if current pixel needs to generate new RIS candidate using temporal data
                    PathReservoir temporalReservoir;
                    bool validTemporalReservoir = true;
                    bool disoccluded;
                    bool incorrectReprojection = false;
                    bool specularOrTransmissiveMaterials = false;
                    var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1 + 1) * params.seed + kCandidateSamples);
                    {
                        float2 motionVector = motionVectors[pixel];
                        int2 prevPixel = pixel;

                        bool foundTemporalSurface = true;

                        if (gEnableTemporalReprojection)
                        {
                            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
                        }

                        if (!isValidScreenRegion(prevPixel) || !foundTemporalSurface) { validTemporalReservoir = false; }

                        PackedHitInfo centralPrimaryHitPacked;
                        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);

                        PackedHitInfo temporalPrimaryHitPacked;
                        ShadingData temporalPrimarySd = getPixelTemporalShadingData(prevPixel, temporalPrimaryHitPacked);
                        if (!isValidPackedHitInfo(temporalPrimaryHitPacked)) { validTemporalReservoir = false; }

                        // bool doTemporalUpdateForDynamicScene = kTemporalUpdateForDynamicScene; // Doesn't seem to be necessary

                        temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];

                        // Disocclusion handling
                        float3 cameraPosW = gScene.camera.getPosition(false);
                        float distanceCurrent = length(abs(centralPrimarySd.posW - cameraPosW));
                        float distancePrev = length(abs(temporalPrimarySd.posW - cameraPosW));
                        disoccluded = distanceCurrent / distancePrev > 1.01f;

                        // Incorrect Reprojection handling
                        float3 temporalColor = temporalReservoir.F * temporalReservoir.weight;
                        if (temporalColor.r + temporalColor.b + temporalColor.g < 0.0001f)
                        {
                            // color.g = 100000.0f;
                            incorrectReprojection = true;
                        }

                        // Handle Specular and Transmissive materials
                        bool transmissiveMaterial = any(temporalPrimarySd.transmission > 0.f);
                        bool specularMaterial = any(temporalPrimarySd.specular > 0.8f);
                        if (transmissiveMaterial || specularMaterial)
                        {
                            specularOrTransmissiveMaterials = true;
                        }
                    }

                    // Calculate chance of generating new samples [0,inf)
                    float candidateChance = 1.f - easingFunction(min(gAdaptiveTemporalHistoryCap, temporalReservoir.M) / gAdaptiveTemporalHistoryCap);
                    if (specularOrTransmissiveMaterials) { candidateChance += 0.5f; }
                    if (!validTemporalReservoir || disoccluded || incorrectReprojection) { candidateChance += 1.f;}
                    candidateChance = min(1.f, max(0.f, candidateChance));

                    targetSampleCount += candidateChance;
                }
            }
        }

        // Determine RIS generation pattern
        uint generationPattern = 0xFFFFFFFF;
        float referenceSampleCount = float(kAdaptiveTileSize) * float(kAdaptiveTileSize);
        // if (targetSampleCount < referenceSampleCount * 0.0625f)
        // {
        //     generationPattern = gPatternOneSixteenth;
        // }
        // else if (targetSampleCount < referenceSampleCount * 0.125f)
        // {
        //     generationPattern = gPatternOneEight;
        // }
        // else if (targetSampleCount < referenceSampleCount * 0.25f)
        // {
        //     generationPattern = gPatternQuarter;
        // }
        // else if (targetSampleCount < referenceSampleCount * 0.5f)
        // {
        //     generationPattern = gPatternHalf;
        // }
        // else if (targetSampleCount < referenceSampleCount * 0.75f)
        // {
        //     generationPattern = gPatternThreeQuarters;
        // }

        if (targetSampleCount < referenceSampleCount * 0.125f)
        {
            generationPattern = gPatternOneSixteenth;
        }
        else if (targetSampleCount < referenceSampleCount * 0.25f)
        {
            generationPattern = gPatternOneEight;
        }
        else if (targetSampleCount < referenceSampleCount * 0.5f)
        {
            generationPattern = gPatternQuarter;
        }
        else if (targetSampleCount < referenceSampleCount * 0.75f)
        {
            generationPattern = gPatternHalf;
        }
        else if (targetSampleCount < referenceSampleCount)
        {
            generationPattern = gPatternThreeQuarters;
        }

        // Fill in local PathID buffers
        uint RISCount = 0;
        uint NonRISCount = 0;
        uint RISIDs[256] = {};    // init to zeros
        uint NonRISIDs[256] = {}; // init to zeros

        for (int i = 0; i < kAdaptiveTileSize; i++)
        {
            for (int j = 0; j < kAdaptiveTileSize; j++)
            {
                const uint2 pixel = startingPixel + uint2(i, j);

                const HitInfo hit = HitInfo(vbuffer[pixel]);
                bool hitSurface = hit.isValid();
                if (!hitSurface && gSampleId == 0)
                    continue;

                uint mortonIdx = interleave_16bit(pixel) % 16;
                uint patternMask = 1 << (15 - mortonIdx);
                bool RISCandidate = ((generationPattern >> gPatternShift) & patternMask) != 0;

                uint pathID = (1U << 24) | (pixel.y << 12) | pixel.x;
                if (RISCandidate)
                {
                    RISIDs[RISCount] = pathID;
                    RISCount++;
                }
                else
                {
                    NonRISIDs[NonRISCount] = pathID;
                    NonRISCount++;
                }
            }
        }

        // Write PathIDs to memory
        uint bufferOffset;
        InterlockedAdd(gRISPathIDs[0], RISCount, bufferOffset);
        bufferOffset++;
        GroupMemoryBarrierWithGroupSync();
        for (int i = 0; i < RISCount; i++)
        {
            gRISPathIDs[bufferOffset + i] = RISIDs[i];
        }

        InterlockedAdd(gNonRISPathIDs[0], NonRISCount, bufferOffset);
        bufferOffset++;
        GroupMemoryBarrierWithGroupSync();
        for (int i = 0; i < NonRISCount; i++)
        {
            gNonRISPathIDs[bufferOffset + i] = NonRISIDs[i];
        }
    }

    void writeBackground(const uint2 pixel, const float3 dir)
    {
        // Evaluate background color for the current pixel.
        float3 color = float3(0.f);
        if (kUseEnvBackground)
        {
            color = gScene.envMap.eval(dir);
        }

        // Write color and denoising guide data for all samples in pixel.
        // For the special case of fixed 1 spp we write the color directly to the output texture.
        {
            outputColor[pixel] = float4(color, 1.f);
        }
    }
};

cbuffer CB
{
    PathGenerator gPathGenerator;
    uint kAdaptiveTileSize;
}

[numthreads(16, 16, 1)]
void main(
    uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 startingPixel = dispatchThreadId.xy * kAdaptiveTileSize;
    gPathGenerator.execute(startingPixel);
}