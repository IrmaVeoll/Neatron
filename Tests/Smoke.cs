using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Neat;
using Xunit;

namespace Tests
{
    public class Smoke
    {
        [Fact]
        public void AsexualReproduction()
        {
            var initialParameters = new NetworkParameters(2, 3, NetworkType.FeedForward)
            {
                InitialConnectionDensity = 0.5f
            };

            var reproductionParameters = new ReproductionParameters()
            {
                WeightMutations = new WeightMutations
                {
                    OverallRouletteWheelShare = 50
                },
                AddConnectionRouletteWheelShare = 5f,
                RemoveConnectionRouletteWheelShare = 5f,
                SplitConnectionRouletteWheelShare = 5f
            };
            
            var neatPopulation = new Population(initialParameters, reproductionParameters);
            var sensorValues = Enumerable.Range(0, initialParameters.SensorCount)
                .Select(_ => (float) new Random().NextDouble()).ToArray();
            
            var genome = neatPopulation.CreateInitialGenome();
            var network = genome.Network;
            for (var i = 0; i < initialParameters.SensorCount; i++)
            {
                network.Sensors[i] = sensorValues[i];
            }
            network.Activate();

            for (var i = 0; i < 50000; i++)
            {
                genome = neatPopulation.Replicator.Reproduce(genome);
                var newNetwork = genome.Network;
                for (var s = 0; s < initialParameters.SensorCount; s++)
                {
                    newNetwork.Sensors[s] = sensorValues[s];
                }
                newNetwork.Activate();
            }
        }

        [Fact]
        public void NetworkReset()
        {
            var initialParameters = new NetworkParameters(10, 5, NetworkType.FeedForward)
            {
                InitialConnectionDensity = 1f
            };

            var reproductionParameters = new ReproductionParameters()
            {
                WeightMutations = new WeightMutations
                {
                    OverallRouletteWheelShare = 0
                },
                AddConnectionRouletteWheelShare = 2f,
                RemoveConnectionRouletteWheelShare = 1f,
                SplitConnectionRouletteWheelShare = 1f
            };
            
            var neatPopulation = new Population(initialParameters, reproductionParameters);

            var genomes = Enumerable.Range(0, 100).Select(_ => neatPopulation.CreateInitialGenome()).ToArray();

            for (var i = 0; i < 500; i++)
            {
                var sensorValues = Enumerable.Range(0, initialParameters.SensorCount)
                    .Select(_ => (float) new Random().NextDouble()).ToArray();

                foreach (var genome in genomes)
                {
                    var network = genome.Network;
                    
                    for (var j = 0; j < initialParameters.SensorCount; j++)
                    {
                        network.Sensors[j] = sensorValues[j];
                    }
                    network.Activate();
                    var effectors = network.Effectors.ToArray();
                    
                    network.Activate();

                    Assert.Equal(effectors, network.Effectors);
                }

                genomes = genomes.Select(g => neatPopulation.Replicator.Reproduce(g)).ToArray();
            }
        }
    }
}