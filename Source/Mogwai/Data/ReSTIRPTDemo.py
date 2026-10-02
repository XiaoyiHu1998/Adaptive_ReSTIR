from falcor import *
import os.path as path
import Data.DataExporter as DE

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


def export_scene_data(scenePath: str, graph):
    m.addGraph(graph)

    m.unloadScene()
    m.loadScene(scenePath)

    print(f"Exporting data from {scenePath}")

    profilerFrameCount = 101
    captureFrameCount = 101

    DE.captureProfilerData(profilerFrameCount, m)
    print(f"captured profiler data for {profilerFrameCount} frames")

    DE.captureFramesPaused(100, m)
    print(f"captured {captureFrameCount} frames")

    DE.captureTiming(m)
    print(f"captured timing data for frames")
    
    m.removeGraph(graph)


graph_ReSTIRPT = render_graph_ReSTIRPT()
# m.addGraph(graph_ReSTIRPT)
# m.loadScene("Arcade/Arcade.pyscene")
# m.loadScene("VeachAjar/VeachAjarAnimated.pyscene")

DE.baseDirectory = "H:/ThesisTestOutputs"

DE.subDirectory = "ReSTIRPTDemoTest_VeachAjar"
export_scene_data("VeachAjar/VeachAjarAnimated.pyscene", graph_ReSTIRPT)

DE.subDirectory = "ReSTIRPTDemoTest_Arcade"
export_scene_data("Arcade/Arcade.pyscene", graph_ReSTIRPT)