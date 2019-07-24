using Neat;
using System.Linq;
using Tests.Fakes;
using Xunit;
using static Neat.NetworkParameters;

namespace Tests.BasicTests
{
    public class BackPropagationTests
    {
        private class DummyNeatChromosomeEncoder : INeatChromosomeEncoder
        {
            public void Encode() {}
        }

        [Fact]
        public void CheckFeedforwardBp()
        {
            const int inputCount = 2;
            const int hiddenCount = 2;
            const int outputCount = 1;

            NetworkConnection[][] GetNeatNetworkConnections(float[] weights)
            {
                //weights layout: i-h weights, i-h biases, h-o weights, h-o biases
                
                var networkConnections = new NetworkConnection[BiasCount + inputCount + outputCount + hiddenCount][];
                
                //bias connections
                networkConnections[0] = new NetworkConnection[hiddenCount + outputCount];

                //h-o biases
                var startIdx = weights.Length - 1;
                for (var j = 0; j < outputCount; j++)
                {
                    networkConnections[0][j] = new NetworkConnection(BiasCount + inputCount + j, weights[startIdx - j]);
                }

                //h-o weights
                startIdx -= outputCount;
                for (var k = 0; k < hiddenCount; k++)
                {
                    networkConnections[networkConnections.Length - 1 - k] = new NetworkConnection[outputCount];

                    for (var j = 0; j < outputCount; j++)
                    {
                        networkConnections[networkConnections.Length - 1 - k][j] =
                            new NetworkConnection(BiasCount + inputCount + j, weights[startIdx - k * outputCount - j]);
                    }
                }

                //i-h biases
                startIdx -= hiddenCount * outputCount;
                for (var k = 0; k < hiddenCount; k++)
                {
                    networkConnections[0][outputCount + k] = new NetworkConnection(networkConnections.Length - 1 - k, weights[startIdx - k]);
                }

                //i-h weights
                startIdx -= hiddenCount;
                for (var i = 0; i < inputCount; i++)
                {
                    networkConnections[i + BiasCount] = new NetworkConnection[hiddenCount];
                    for (var k = 0; k < hiddenCount; k++)
                    {
                        networkConnections[i + BiasCount][k] =
                            new NetworkConnection(networkConnections.Length - 1 - k, weights[startIdx - i * hiddenCount - k]);
                    }
                }

                for (var j = 0; j < outputCount; j++)
                {
                    networkConnections[BiasCount + inputCount + j] = new NetworkConnection[0];
                }
                
                return networkConnections;
            }

            var sampleNetwork = new SampleNetwork(inputCount, hiddenCount, outputCount);
            var neatNetwork = new Network(GetNeatNetworkConnections(sampleNetwork.GetWeights()), 
                inputCount + BiasCount, 
                outputCount, 
                new DummyNeatChromosomeEncoder());

            var sampleNetworkOutput = sampleNetwork.ComputeOutputs(new[] { 0f, 1f });
            
            neatNetwork.Sensors[0] = 1f;
            neatNetwork.Sensors[1] = 0f;
            neatNetwork.Activate();

            var neatNetworkOutputBeforeTraining = neatNetwork.Effectors.ToArray();

            //the sample network and the NEAT network are identical
            Assert.Equal(sampleNetworkOutput, neatNetworkOutputBeforeTraining);

            sampleNetwork.Train(new[] {new[] {0f, 1f, 1f}}, 0.3f);
            neatNetwork.ActivateAndTrain(new[] {1f, 0f, 1f}, 0.3f);

            //NEAT network effector values have not changed after training before Activate call
            Assert.Equal(neatNetworkOutputBeforeTraining, neatNetwork.Effectors.ToArray());

            sampleNetworkOutput = sampleNetwork.ComputeOutputs(new[] { 0f, 1f });
            neatNetwork.Activate();
            
            //backpropagation works fine
            Assert.Equal(sampleNetworkOutput, neatNetwork.Effectors.ToArray());
            Assert.NotEqual(neatNetworkOutputBeforeTraining, neatNetwork.Effectors.ToArray());
        }
    }
}