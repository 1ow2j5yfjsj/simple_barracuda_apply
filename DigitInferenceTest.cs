// Just put it in the GameObject

using System;

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

using Unity.Barracuda;
using Unity.MLAgents.Sensors;


public class DigitInferenceTest : MonoBehaviour
{
    public NNModel modelFile;

    private IWorker worker;
    Rigidbody m_AgentRb;

    // Start is called before the first frame update
    void Start()
    {
        var model = ModelLoader.Load(this.modelFile);
        this.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

    }

    public void TestInference(byte[] byteArray)
    {
        
        
    }


    // Update is called once per frame
    void Update()
    {
        // m_Texture = new Texture2D(84, 84, TextureFormat.RGB24, false);
        
        // var oldRec = cm.rect;
        // cm.rect = new Rect(0f, 0f, 1f, 1f);
        // var depth = 24;
        // var format = RenderTextureFormat.Default;
        // var readWrite = RenderTextureReadWrite.Default;

        // var tempRt = RenderTexture.GetTemporary(84, 84, depth, format, readWrite);

        // var prevActiveRt = RenderTexture.active;
        // var prevCameraRt = cm.targetTexture;

        // // render to offscreen texture (readonly from CPU side)
        // RenderTexture.active = tempRt;
        // cm.targetTexture = tempRt;

        // cm.Render();

        // m_Texture.ReadPixels(new Rect(0, 0, m_Texture.width, m_Texture.height), 0, 0);
 
        // cm.targetTexture = prevCameraRt;
        // cm.rect = oldRec;
        // RenderTexture.active = prevActiveRt;
        // RenderTexture.ReleaseTemporary(tempRt);

        // //var compressed = m_Texture.EncodeToPNG();
        // var compressed = m_Texture.GetRawTextureData();

        float[] input_array = new float[1]{1f};

        //Debug.Log(string.Format("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}", scale_float[0], scale_float[1], scale_float[2], scale_float[3], scale_float[4], scale_float[5], scale_float[6], scale_float[7], scale_float[8]));

        var input_tensor = new Tensor(1, 1, 1, 1, input_array);

        var inputs = new Dictionary<string, Tensor>();
        inputs.Add("obs_0", input_tensor);

        worker.Execute(inputs);
        
        var output = worker.PeekOutput("discrete_actions");
        
        //Debug.Log(output[0]);
        Debug.Log(string.Format("{0:f6}, {1:f6}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}", output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9]));

    }
}
