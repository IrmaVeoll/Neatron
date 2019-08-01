using Neat;
using System;
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
            public void Encode() { }
        }

        [Fact]
        public void CheckFeedforwardBp()
        {
            const int inputCount = 2;
            const int hiddenCount = 2;
            const int outputCount = 1;

            var referenceNetwork = new ReferenceFFNetwork(inputCount, hiddenCount, outputCount);
            var neatNetwork = new Network(GetNeatNetworkConnections(referenceNetwork.GetWeights(), inputCount, hiddenCount, outputCount),
                inputCount + BiasCount,
                outputCount,
                new DummyNeatChromosomeEncoder());

            var sampleNetworkOutput = referenceNetwork.ComputeOutputs(new[] { 0f, 1f });

            neatNetwork.Sensors[0] = 1f;
            neatNetwork.Sensors[1] = 0f;
            neatNetwork.Activate();

            var neatNetworkOutputBeforeTraining = neatNetwork.Effectors.ToArray();

            //the sample network and the NEAT network are identical
            Assert.Equal(sampleNetworkOutput, neatNetworkOutputBeforeTraining);

            referenceNetwork.Train(new[] { new[] { 0f, 1f, 1f } }, 0.3f);
            neatNetwork.ActivateAndTrain(new[] { 1f, 0f, 1f }, 0.3f);

            //NEAT network effector values have not changed after training before Activate call
            Assert.Equal(neatNetworkOutputBeforeTraining, neatNetwork.Effectors.ToArray());

            sampleNetworkOutput = referenceNetwork.ComputeOutputs(new[] { 0f, 1f });
            neatNetwork.Activate();

            //backpropagation works fine
            Assert.Equal(sampleNetworkOutput, neatNetwork.Effectors.ToArray());
            Assert.NotEqual(neatNetworkOutputBeforeTraining, neatNetwork.Effectors.ToArray());
        }

        [Fact]
        public void CheckAccuracy()
        {
            Random rand = new Random(31415);

            float[] GetSample()
            {
                var s = new float[3];
                s[0] = rand.Next(2);
                s[1] = rand.Next(2);
                s[2] = ((int)s[0]) ^ ((int)s[1]);
                return s;
            }

            float Normalize(float value)
            {
                return value * 2f - 1;
            }

            const int inputCount = 2;
            const int hiddenCount = 5;
            const int outputCount = 1;

            var referenceNetwork = new ReferenceFFNetwork(inputCount, hiddenCount, outputCount);
            var neatNetwork = new Network(GetNeatNetworkConnections(referenceNetwork.GetWeights(), inputCount, hiddenCount, outputCount),
                inputCount + BiasCount,
                outputCount,
                new DummyNeatChromosomeEncoder());

            var sample = GetSample();
            neatNetwork.Sensors[0] = Normalize(sample[0]);
            neatNetwork.Sensors[1] = Normalize(sample[1]);
            neatNetwork.Activate();

            var err = MathF.Abs(Normalize(neatNetwork.Effectors[0]) - sample[2]);

            for (var i = 0; i < 100; i++)
            {
                var s = GetSample();
                neatNetwork.ActivateAndTrain(s, 0.1f);
                referenceNetwork.Train(new[] { s }, 0.1f);
            }

            neatNetwork.Sensors[0] = Normalize(sample[0]);
            neatNetwork.Sensors[1] = Normalize(sample[1]);
            neatNetwork.Activate();
            var referenceNetworkOutput = referenceNetwork.ComputeOutputs(sample.Take(2).ToArray());

            Assert.Equal(referenceNetworkOutput[0], neatNetwork.Effectors[0], 7);
            Assert.True(err > MathF.Abs(Normalize(neatNetwork.Effectors[0]) - sample[2]));
        }

        private static NetworkConnection[][] GetNeatNetworkConnections(float[] weights, int inputCount, int hiddenCount, int outputCount)
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
    }
}