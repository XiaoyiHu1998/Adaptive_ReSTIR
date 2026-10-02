from falcor import *
import os.path as path
import Data.DataExporter as DE
import Data.Export.ExportConfig as cfg

def render_graph_ReSTIRPT():
    g = RenderGraph("ReSTIRPTPass")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ReSTIRPTPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")
    loadRenderPassLibrary("ErrorMeasurePass.dll")
    loadRenderPassLibrary("ImageLoader.dll")

    ReSTIRGIPlusPass = createPass("ReSTIRPTPass", {'samplesPerPixel': 1})
    g.addPass(ReSTIRGIPlusPass, "ReSTIRPTPass")
    VBufferRT = createPass("VBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'texLOD': TexLODMode.Mip0, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': False, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass")    
    g.addPass(ScreenSpaceReSTIRPass, "ScreenSpaceReSTIRPass")
    
    g.addEdge("VBufferRT.vbuffer", "ReSTIRPTPass.vbuffer")   
    g.addEdge("VBufferRT.mvec", "ReSTIRPTPass.motionVectors")    
    
    g.addEdge("VBufferRT.vbuffer", "ScreenSpaceReSTIRPass.vbuffer")   
    g.addEdge("VBufferRT.mvec", "ScreenSpaceReSTIRPass.motionVectors")    
    g.addEdge("ScreenSpaceReSTIRPass.color", "ReSTIRPTPass.directLighting")    
    
    g.addEdge("ReSTIRPTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    g.markOutput("AccumulatePass.output")  

    return g


def export_performance_data(scenePath: str, frameCount: int):
    m.loadScene(scenePath)

    DE.captureProfilerData(frameCount, m)
    print(f"captured profiler data for {frameCount} frames")

    # DE.captureTiming(m)
    # print(f"captured timing data for frames")

    m.unloadScene()


def export_frame_data(scenePath: str, frameCount: int):
    m.loadScene(scenePath)

    DE.captureFramesPaused(frameCount, m)
    print(f"captured {frameCount} frames")

    m.unloadScene()


# Setup Falcor renderer
graph_ReSTIRPT = render_graph_ReSTIRPT()
m.addGraph(graph_ReSTIRPT)

m.profiler.enabled = False
m.profiler.paused = True
# m.clock.framerate = 30
# m.clock.pause()

# Export config
DE.baseDirectory = cfg.baseDirectory
DE.sceneName = cfg.ArcadeName
DE.runName = cfg.activeReSTIRScheme + "_" + __file__.split(".")[0].split("_")[-1]

# Export Data
export_frame_data(cfg.ArcadePath, cfg.frameDataFrames)
exit()
