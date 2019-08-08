using System;
using Neat;
using System.Linq;
using Tests.Fakes;
using Xunit;
using static Neat.NetworkParameters;
using RecurrentNetwork = Neat.RecurrentNetwork;

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

        [Fact]
        public void CheckRecurrentBp()
        {
            float GetSampleSetError(float[][] floats, RecurrentNetwork network1)
            {
                var error = 0f;
                foreach (var sample in floats)
                {
                    network1.Sensors[0] = sample[0];
                    network1.Activate();
                    error += Math.Abs(sample[1] - network1.Effectors[0]);
                }

                return error;
            }

            void FlushNetworkState(RecurrentNetwork recurrentNetwork)
            {
                for (var i = 0; i < 3; i++)
                {
                    recurrentNetwork.Sensors[0] = 0f;
                    recurrentNetwork.Activate();
                }
            }

            float[][] GenerateSampleDataset(int size)
            {
                var result = new float[size][];
                var previous = 0f;
                var current = 0f;
                for (var i = 0; i < size; i++)
                {
                    result[i] = new float[2];
                    result[i][1] = current + previous; // previous inputs sum for output
                    previous = current;
                    current = Neat.Utils.RandomSource.Range(-0.5f, 0.5f);
                    result[i][0] = current;
                }

                return result;
            }

            // Arrange
            var connections = new NetworkConnection[BiasCount + 3][]; // 0 : bias, 1: input, 2: output, 3: hidden
            connections[0] = new NetworkConnection[1];
            connections[0][0] = new NetworkConnection(2, 1f); // bias to output
            connections[1] = new NetworkConnection[2];
            connections[1][0] = new NetworkConnection(2, 2f); // input to output
            connections[1][1] = new NetworkConnection(3, -1f); // input to hidden
            connections[2] = new NetworkConnection[0]; // no output self-links
            connections[3] = new NetworkConnection[1];
            connections[3][0] = new NetworkConnection(2, 0f); // hidden to output
            var network = new RecurrentNetwork(connections, 2, 1, new DummyNeatChromosomeEncoder());

            // Act
            FlushNetworkState(network);
            var sampleDataSet = GenerateSampleDataset(10);
            var beforeTrainError = GetSampleSetError(sampleDataSet, network);

            for (var i = 0; i < 20; i++)
            {
                // flush network state
                FlushNetworkState(network);

                network.Train(sampleDataSet, 0.1f);
            }

            FlushNetworkState(network);
            var afterTrainError = GetSampleSetError(sampleDataSet, network);

            for (var i = 0; i < 20; i++)
            {
                // flush network state
                FlushNetworkState(network);

                network.Train(sampleDataSet, 0.1f);
            }

            FlushNetworkState(network);
            var afterSecondTrainError = GetSampleSetError(sampleDataSet, network);

            // Assert
            Assert.True(afterTrainError < beforeTrainError, "Training failed!");
            Assert.True(afterSecondTrainError < afterTrainError, "Second training failed!");
        }


        [Fact]
        public void CheckBpWithSampleNetwork()
        {
            float GetWeightDifference(NetworkConnection[][] first, NetworkConnection[][] second)
            {
                var difference = 0f;
                for (var i = 0 ; i < first.Length; i++)
                for (var j = 0; j < first[i].Length; j++)
                    difference += Math.Abs(first[i][j].Weight - second[i][j].Weight);

                return difference;
            }

            float[][] GenerateSampleDataset(RecurrentNetwork sample, int size)
            {
                var result = new float[size][];
                for (var i = 0; i < size; i++)
                {
                    result[i] = new float[3];
                    var input1 = Neat.Utils.RandomSource.Range(-1f, 1f);
                    var input2 = Neat.Utils.RandomSource.Range(-1f, 1f);
                    sample.Sensors[0] = input1;
                    sample.Sensors[1] = input2;
                    sample.Activate();
                    result[i][0] = input1;
                    result[i][1] = input2;
                    result[i][2] = sample.Effectors[0];
                }

                return result;
            }

            // Arrange
            var sampleConnections =
                new NetworkConnection[BiasCount + 4][]; // 0 : bias, 1, 2: inputs, 3: output, 4: hidden
            sampleConnections[0] = new NetworkConnection[1];
            sampleConnections[0][0] = new NetworkConnection(3, 1f); // bias to output
            sampleConnections[1] = new NetworkConnection[1];
            sampleConnections[1][0] = new NetworkConnection(4, 0.5f); // input to hidden
            sampleConnections[2] = new NetworkConnection[1];
            sampleConnections[2][0] = new NetworkConnection(4, -0.5f); // input to hidden
            sampleConnections[3] = new NetworkConnection[1];
            sampleConnections[3][0] = new NetworkConnection(4, 0.2f); // output to hidden
            sampleConnections[4] = new NetworkConnection[1];
            sampleConnections[4][0] = new NetworkConnection(3, -1f); // hidden to output
            var sampleNetwork = new RecurrentNetwork(sampleConnections, 3, 1, new DummyNeatChromosomeEncoder());
            var targetConnections =
                new NetworkConnection[BiasCount + 4][]; // 0 : bias, 1, 2: inputs, 3: output, 4: hidden
            targetConnections[0] = new NetworkConnection[1];
            targetConnections[0][0] = new NetworkConnection(3, 0.8f); // bias to output
            targetConnections[1] = new NetworkConnection[1];
            targetConnections[1][0] = new NetworkConnection(4, 0.7f); // input to hidden
            targetConnections[2] = new NetworkConnection[1];
            targetConnections[2][0] = new NetworkConnection(4, -0.3f); // input to hidden
            targetConnections[3] = new NetworkConnection[1];
            targetConnections[3][0] = new NetworkConnection(4, 0.1f); // output to hidden
            targetConnections[4] = new NetworkConnection[1];
            targetConnections[4][0] = new NetworkConnection(3, -0.8f); // hidden to output
            var targetNetwork = new RecurrentNetwork(targetConnections, 3, 1, new DummyNeatChromosomeEncoder());

            // Act
            var beforeTrainDifference = GetWeightDifference(sampleConnections, targetConnections);

            for (var i = 0; i < 5000; i++)
            {
                targetNetwork.Train(GenerateSampleDataset(sampleNetwork, 5), 0.5f);
            }

            var afterTrainDifference = GetWeightDifference(sampleConnections, targetConnections);

            // Assert
            Assert.True(afterTrainDifference < beforeTrainDifference, "Training failed!");
        }
    }
}