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
    RWBuffer<uint> gPixelCandidateStatus;
    RWBuffer<uint> gNonRISFrameCounters;
    RWBuffer<float> gDuplicationMap;

    int gSampleId;
    bool gEnableTemporalReprojection;
    float gAdaptiveTemporalHistoryCap;
    float gAdaptiveMinPerPixelRISRate;
    float gAdaptiveMaxPerPixelRISRate;
    bool gEnableDuplicationMapping;

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
    void execute(const uint2 tileID, const uint threadIdx)
    {
        // Map thread to pixel based on Morton order within tile.
        // The tiles themselves are enumerated in scanline order on screen.
        const uint2 tileOffset = tileID << kScreenTileBits;            // Tile offset in pixels.
        const uint2 pixel = deinterleave_8bit(threadIdx) + tileOffset; // Assumes 16x16 tile or smaller. A host-side assert checks this assumption.

        // Process the pixel to determine which queue to use.
        // If we don't hit any surfaces or volumes, then the background will be evaluated and written out directly.
        Ray cameraRay;
        bool useGeneralQueue = false; // Does not handle no hit only

        // Note: Do not terminate threads for out-of-bounds pixels because we need all threads active for the prefix sum pass below.
        if (all(pixel < params.frameDim))
        {
            // Determine number of samples at the current pixel.
            // This is either a fixed number or loaded from the sample count texture.
            // TODO: We may want to use a nearest sampler to allow the texture to be of arbitrary dimension.

            // Compute the primary ray.
            // TODO: Support depth-of-field using computeRayThinlens, if aperture size > 0.
            cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);

            // Load the primary hit from the V-buffer.
            const HitInfo hit = HitInfo(vbuffer[pixel]);
            bool hitSurface = hit.isValid();

            if (!hitSurface && gSampleId == 0)
            {
                // Write background pixels.
                writeBackground(pixel, cameraRay.dir);
                return;
            }

            // Determine if current pixel needs to generate new RIS candidate using temporal data
            PathReservoir temporalReservoir;
            uint prevPixelOffset;
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

                prevPixelOffset = params.getReservoirOffset(prevPixel);
                temporalReservoir = temporalReservoirs[prevPixelOffset];

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

            static groupshared uint reservoirConfidenceWeights[256];
            reservoirConfidenceWeights[threadIdx % 256] = temporalReservoir.rcRandomSeed;
            GroupMemoryBarrierWithGroupSync();

            float duplicationScore = -1.f / 255.f;
            for (int i = 0; i < 256; i++)
            {
                if (temporalReservoir.rcRandomSeed == reservoirConfidenceWeights[i]) { duplicationScore += 1.f / 255.f; }
            }
            gDuplicationMap[pixel.x + pixel.y * params.frameDim.x] = duplicationScore;

            // Check if pixel has gone without new sample for too long
            uint framesSinceRISCandidate = gNonRISFrameCounters[pixel.x + pixel.y * params.frameDim.x];
            bool tooLongWithoutRISCandidate = framesSinceRISCandidate >= 15;
            // bool tooLongWithoutRISCandidate = false;

            // Calculate chance of generating new samples [0,inf)
            float candidateChance = 1.f - easingFunction(min(gAdaptiveTemporalHistoryCap, temporalReservoir.M) / gAdaptiveTemporalHistoryCap);
            if (specularOrTransmissiveMaterials) { candidateChance = 0.5f; }
            if (!validTemporalReservoir || disoccluded || incorrectReprojection || tooLongWithoutRISCandidate) { candidateChance = 1.0f; }
            candidateChance += duplicationScore;
            candidateChance = min(gAdaptiveMaxPerPixelRISRate, max(gAdaptiveMinPerPixelRISRate, candidateChance));

            // candidateChance / 1.0f gives the amount of guaranteed new samples and candidateChance % 1.0f gives chance of an additional sample
            uint sampleCount = uint(candidateChance / 1.f);
            if (sampleNext1D(sg) <= (candidateChance % 1.f))
                sampleCount++;
            sampleCount = min(1, sampleCount);
            bool generateRIS = sampleCount > 0;

            // Update NonRISFrameTimers
            gNonRISFrameCounters[pixel.x + pixel.y * params.frameDim.x] = generateRIS ? 0 : framesSinceRISCandidate + 1;

            // Write pathID to gRISPathIDs buffer for each RIS candidate pixel using prefix sum
            bool RISPixel = generateRIS;
            uint prefixSum = WavePrefixCountBits(RISPixel);
            uint generateCount = WaveActiveCountBits(RISPixel);

            uint bufferOffset;
            if (WaveIsFirstLane())
            {
                InterlockedAdd(gRISPathIDs[0], generateCount, bufferOffset);
            }
            bufferOffset = WaveReadLaneFirst(bufferOffset) + 1;
            bufferOffset += prefixSum;
            GroupMemoryBarrierWithGroupSync();
            if (RISPixel)
            {
                gRISPathIDs[bufferOffset] = (1U << 24) | (pixel.y << 12) | pixel.x; // pathID stores 8 bit sample count, 12 bit y, 12 bit x in this order
            }

            GroupMemoryBarrierWithGroupSync();
            gPixelCandidateStatus[pixel.x + pixel.y * params.frameDim.x] = RISPixel ? 1u : 0u;
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
}

// TODO: Replace by compile-time uint2 constant when it works in Slang.
// See related issue https://gitlab-master.nvidia.com/nvresearch-gfx/Tools/slang/-/issues/87
[numthreads(256 /* kScreenTileDim.x * kScreenTileDim.y */, 1, 1)]
void main(
    uint3 groupID: SV_GroupID,
    uint3 groupThreadID: SV_GroupThreadID)
{
    gPathGenerator.execute(groupID.xy, groupThreadID.x);
}