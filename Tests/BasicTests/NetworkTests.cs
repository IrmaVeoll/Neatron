using System.Linq;
using Neat;
using Tests.Fakes;
using Xunit;

namespace Tests.BasicTests
{
    public class NetworkTests
    {
        [Fact]
        public void SimpleNetworkCheck()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                for (var i = 0; i < 10; i++)
                {
                    source.PushNextFloats(0.6f); // identity weights and density chances
                }

                source.GetParanoid();

                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var genome = new Genome(parameters, tracker, false);

                var network = new RecurrentNetwork();

                #region Network initialization

                network.AddNeuron(0, 0);
                network.AddNeuron(1, 0);
                network.AddNeuron(2, 0);
                network.AddNeuron(3, 0);
                network.AddNeuron(4, 1);
                network.AddConnection(0, 4, 1f);
                network.AddConnection(1, 4, 1f);
                network.AddConnection(2, 4, 1f);
                network.AddConnection(3, 4, 1f);
                network.AddConnection(4, 4, 1f);

                #endregion

                float[] result = null;
                for (var i = 0; i < 5; i++)
                {
                    for (var j = 0; j < 3; j++)
                    {
                        genome.Network.Sensors[j] = 0.5f;
                    }

                    genome.Network.Activate();

                    result = network.Activate(new[] {1f, 0.5f, 0.5f, 0.5f});
                }

                Assert.Equal(result[0], genome.Network.Effectors[0]);
            }
        }

        [Fact]
        public void LayeredNetworkCheck()
        {
            //TODO fix our RNN or delete this test
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);
            genome.SplitConnection(0);
            genome.SplitConnection(2);
            genome.SplitConnection(genome.NeatChromosome.Single(g => g.SourceId == 5).Id);
            while(genome.VacantConnectionCount > 0)
                genome.AddConnection(Neat.Utils.RandomSource.Next(genome.VacantConnectionCount));

            var network = new RecurrentNetwork();
            
            void AddConnectionAndCopyWeight(int sourceId, int targetId)
            {
                var weight = genome.NeatChromosome.Single(g => g.SourceId == sourceId && g.TargetId == targetId).Weight;
                network.AddConnection(sourceId, targetId, weight);
            }

            #region Network initialization

            network.AddNeuron(0, 0);
            network.AddNeuron(1, 0);
            network.AddNeuron(2, 0);
            network.AddNeuron(3, 0);
            network.AddNeuron(5, 1);
            network.AddNeuron(6, 1);
            network.AddNeuron(7, 2);
            network.AddNeuron(4, 3);
            for(var i = 0; i < 4; i++)
                for(var j = 4; j < 8; j++)
                    AddConnectionAndCopyWeight(i, j);
            for(var i = 4; i < 8; i++)
                for(var j = 4; j < 8; j++)
                    AddConnectionAndCopyWeight(i, j);

            Assert.Equal(network.ConnectionsCount, genome.NeatChromosome.Count);
            #endregion

            float[] result = null;
            for (var i = 0; i < 5; i++)
            {
                var values = new[]
                {
                    1f, 
                    Neat.Utils.RandomSource.Next(), 
                    Neat.Utils.RandomSource.Next(), 
                    Neat.Utils.RandomSource.Next()
                };
                
                for (var j = 0; j < 3; j++)
                {
                    genome.Network.Sensors[j] = values[j+1];
                }

                genome.Network.Activate();

                result = network.Activate(values);
            }

            Assert.Equal(result[0], genome.Network.Effectors[0]);
        }
    }
}