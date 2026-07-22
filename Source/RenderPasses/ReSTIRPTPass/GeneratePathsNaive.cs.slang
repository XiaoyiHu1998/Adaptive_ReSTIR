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
    RWTexture2D<float4> outputColor;  ///< Output color buffer if kSamplesPerPixel == 1.
    RWBuffer<uint> pathIDs;

    int gSampleId;
    // uint gSamplingRateRIS; // Shading rates : [0: full, 1: three - quarters, 2: half, 3: quarter]
    uint gPatternNumber; // Determine which pattern frame
    uint gPatterns; // Uint encoded with all pixel states for generation patterns across 4 frames in the 16 least significant bits

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    // static bool kUseEnvBackground = USE_ENV_BACKGROUND;
    bool kUseEnvBackground;

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
                // return;
            }

            // Determine if RIS candidate needs to be generated
            uint mortonIdx = threadIdx % 4;                                        // pixel index in 2x2 morton-order grid
            uint patternsMask = (1 << (3 - mortonIdx)) << ((3 - gPatternNumber) * 4); // mask used to select pixel from patterns

            // Write pathID to pathIDs buffer for each RIS candidate using prefix sum
            bool generateCandidate = (gPatterns & patternsMask) != 0;
            uint prefixSum = WavePrefixCountBits(generateCandidate);
            uint generateCount = WaveActiveCountBits(generateCandidate);
            
            // For Debugging
            // if (!hitSurface && gSampleId == 0)
            // {
            //     // Write background pixels.
            //     if (generateCandidate)
            //     {
            //         writeBackground(pixel, cameraRay.dir);
            //     }
            //     else
            //     {
            //         outputColor[pixel] = float4(1.0f, 0.0f, 1.0f, 1.f);
            //     }
            // }

            uint bufferOffset;
            if (WaveIsFirstLane())
            {
                InterlockedAdd(pathIDs[0], generateCount, bufferOffset); // TODO: Remember to reset to 0 in later shader passes
            }
            bufferOffset = WaveReadLaneFirst(bufferOffset) + 1;
            bufferOffset += prefixSum;

            GroupMemoryBarrierWithGroupSync();
            if (generateCandidate)
            {
                pathIDs[bufferOffset] = (pixel.y << 12) | pixel.x;
            }
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