using System.Collections.Generic;

namespace Neat
{
    public interface INetwork
    {
        IList<float> Sensors { get; }
        IReadOnlyList<float> Effectors { get; }
        void Activate();
        void Train(float[][] samples, float learningRate = 0.01f, float l1Ratio = 0, float l2Ratio = 0);
    }
}