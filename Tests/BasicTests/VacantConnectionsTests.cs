using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using FluentAssertions;
using FluentAssertions.Execution;
using Neat;
using Neat.Utils;
using Xunit;

namespace Tests.BasicTests
{
    public class VacantConnectionsTests
    {
        [Fact]
        public void SanityCheck()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);

            // Act

            // Assert
            Assert.Equal(0, genome.VacantConnectionCount);

            // Act
            genome.RemoveConnection(1);

            // Assert
            Assert.Equal(1, genome.VacantConnectionCount);

            // Act
            genome.AddConnection(0);

            // Assert
            Assert.Equal(0, genome.VacantConnectionCount);
        }

        [Fact]
        public void SplitConnectionCheck()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);

            // Act
            genome.SplitConnection(0); // 0 -> 4 into 0 -> 5 -> 4

            // Assert
            Assert.Equal(6, genome.VacantConnectionCount);

            // Act
            genome.SplitConnection(2); // 2 -> 4 into 2 -> 6 -> 4

            // Assert
            Assert.Equal(14, genome.VacantConnectionCount);
        }

        [Fact]
        public void SplitConnectionTwoLayer()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);

            // Act
            genome.SplitConnection(0); // 0 -> 4 into 0 -> 5 -> 4
            genome.SplitConnection(5); // 0 -> 5 -> 4 into 0 -> 6 -> 5 -> 4

            // Assert
            Assert.Equal(14, genome.VacantConnectionCount);
        }

        [Fact]
        public void AddConnectionsOneByOneRecurrent()
        {
            (Genome genome, List<string> currentConnections, List<string> vacantConnections) CreateGenome()
            {
                // ReSharper disable once RedundantArgumentDefaultValue
                var parameters = new NetworkParameters(3, 1, NetworkType.Recurrent) {InitialConnectionDensity = 1f};
                var tracker = new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var genome = new Genome(parameters, tracker, false);
                var currentConnections = new List<string>
                {
                    "0b->4e", "1s->4e", "2s->4e", "3s->4e", "4e->4e"
                };
                var vacantConnections = new List<string>();

                AssertCurrentConnections(genome, currentConnections);
                AssertVacantConnections(genome, vacantConnections);

                // Act
                genome.SplitConnection(0);
                currentConnections.ChangeCollection(new[] {"0b->4e"}, new[] {"0b->5h", "5h->4h"});
                vacantConnections.AddRange(new[] {"0b->4e", "1s->5h", "2s->5h", "3s->5h", "5h->5h", "4e->5h"});
                AssertCurrentConnections(genome, currentConnections);
                AssertVacantConnections(genome, vacantConnections);

                genome.SplitConnection(5);
                currentConnections.ChangeCollection(new[] {"0b->5h"}, new[] {"0b->6h", "6h->5h"});
                vacantConnections.AddRange(new[] {"0b->5h", "1s->6h", "2s->6h", "3s->6h", "6h->6h", "5h->6h", "6h->4e", "4e->6h"});
                AssertCurrentConnections(genome, currentConnections);
                AssertVacantConnections(genome, vacantConnections);

                genome.SplitConnection(1);
                currentConnections.ChangeCollection(new[] {"2s->4e"}, new[] {"2s->7h", "7h->4e"});
                vacantConnections.AddRange(new[] {"2s->4e", "0b->7h", "1s->7h", "3s->7h", "7h->7h", "5h->7h", "6h->7h", "7h->6h", "7h->5h", "4e->7h"});
                AssertCurrentConnections(genome, currentConnections);
                AssertVacantConnections(genome, vacantConnections);

                vacantConnections.Should().HaveCount(24);

                return (genome, currentConnections, vacantConnections);
            }

            var expectedAddStraightOrder = new[]
            {
                "0b->4e", "2s->4e", // outerConnections 
                "0b->5h", "1s->5h", "2s->5h", "3s->5h", "4e->5h", "5h->5h", "7h->5h", // to 5h
                "1s->6h", "2s->6h", "3s->6h", "4e->6h", "5h->6h", "6h->6h", "7h->6h", // to 6h
                "0b->7h", "1s->7h", "3s->7h", "4e->7h", "5h->7h", "6h->7h", "7h->7h", // to 7h
                "6h->4e" // from hidden to effectors
            };

            TestAddConnections(CreateGenome, expectedAddStraightOrder);
        }

        [Fact]
        public void AddConnectionsOneByOneFeedforward()
        {
            // ReSharper disable once RedundantArgumentDefaultValue
            var parameters = new NetworkParameters(3, 1, NetworkType.FeedForward) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);
            var currentConnections = new List<string>
            {
                "0b->4e", "1s->4e", "2s->4e", "3s->4e"
            };

            AssertCurrentConnections(genome, currentConnections);

            genome.SplitConnection(0); // 0 -> 4 into 0 -> 5 -> 4
            currentConnections.ChangeCollection(new[] {"0b->4e"}, new[] {"0b->5h", "5h->4e"});
            AssertCurrentConnections(genome, currentConnections);

            genome.SplitConnection(4); // 0 -> 5 -> 4 into 0 -> 6 -> 5 -> 4
            currentConnections.ChangeCollection(new[] {"0b->5h"}, new[] {"0b->6h", "6h->5h"});
            AssertCurrentConnections(genome, currentConnections);

            genome.SplitConnection(1); // 2 -> 4 into 2 -> 7 -> 4
            currentConnections.ChangeCollection(new[] {"2s->4e"}, new[] {"2s->7h", "7h->4e"});
            AssertCurrentConnections(genome, currentConnections);
        
            var initialLayers = genome.NetworkTopology.LayerRanges.ToArray();
            var initialLinksCount = genome.NetworkTopology.Links.Sum(l => l.Count);
            
            genome.AddConnection(15); // 5h->7h
            AssertNoChanges();
            
            genome.AddConnection(6); // 7h->5h
            AssertNoChanges();
            
            genome.AddConnection(9); // 5h->6h
            AssertNoChanges();
            
            genome.AddConnection(9); // 7h->6h
            AssertNoChanges();
            
            
            genome.AddConnection(12); // 6h->7h
            genome.NetworkTopology.Links.Sum(l => l.Count).Should().Be(initialLinksCount + 1, "added allowed connection");
            genome.NetworkTopology.LayerRanges.Should().BeEquivalentTo(initialLayers, "expected no changes in layer structure");
            

            void AssertNoChanges()
            {
                genome.NetworkTopology.Links.Sum(l => l.Count).Should().Be(initialLinksCount, "expected no changes after adding recurrent connections");
                genome.NetworkTopology.LayerRanges.Should().BeEquivalentTo(initialLayers, "expected no changes after adding recurrent connections");   
            }
        }

        private void TestAddConnections(Func<(Genome, List<string> currentConnections, List<string> vacantConnections)> createGenomeFunc,
            string[] expectedAddOrder)
        {
            Genome genome;
            List<string> currentConnections;
            List<string> vacantConnections;

            void RecreateGenome()
            {
                (genome, currentConnections, vacantConnections) = createGenomeFunc();
            }

            RecreateGenome();
            var allPossibleConnections = vacantConnections.Concat(currentConnections).ToList();
            expectedAddOrder.Should().BeEquivalentTo(vacantConnections, "we expect all vacant connections to be tested");

            // check straight add order
            CheckConnectionOrder(genome, expectedAddOrder, currentConnections, reversed: false);
            genome.VacantConnectionCount.Should().Be(0, "all expected connection have already been added");

            // check reversed add order
            RecreateGenome();
            CheckConnectionOrder(genome, expectedAddOrder.Reverse().ToArray(), currentConnections, reversed: true);
            genome.VacantConnectionCount.Should().Be(0, "all expected connection have already been added");

            RecreateGenome();
            // check random add order
            for (var i = genome.VacantConnectionCount; i > 0; i--)
            {
                genome.AddConnection(RandomSource.Next(i));
                genome.VacantConnectionCount.Should().Be(i - 1, "vacant connections count should decrease one by one");
            }

            AssertCurrentConnections(genome, allPossibleConnections);
        }

        private static void AssertCurrentConnections(Genome genome, IEnumerable<string> expectedConnections)
        {
            var expectedNeuronConnections = expectedConnections.Select(ConnectionNeuronsFromString)
                .OrderBy(x => x.SourceId).ThenBy(x => x.TargetId).ToList();

            var actualNeuronConnections = genome.NeatChromosome.Select(gene => gene.ConnectionNeurons)
                .OrderBy(x => x.SourceId).ThenBy(x => x.TargetId).ToList();

            try
            {
                actualNeuronConnections.Should().BeEquivalentTo(expectedNeuronConnections);
            }
            catch (Exception e)
            {
                var failMessage = new StringBuilder();
                failMessage.AppendLine(e.Message);
                failMessage.AppendLine($"expected connections: {string.Join(", ", expectedNeuronConnections)}");
                failMessage.AppendLine($"actual   connections: {string.Join(", ", actualNeuronConnections)}");
                throw new AssertionFailedException(failMessage.ToString());
            }
        }

        private static void CheckConnectionOrder(Genome testingGenome, string[] expectedOrder, List<string> currentConnections, bool reversed)
        {
            var expectedVacantCollections = new List<string>(expectedOrder);
            AssertCurrentConnections(testingGenome, currentConnections);

            var curConnections = new List<string>(currentConnections);
            foreach (var expectedNewConnection in expectedOrder)
            {
                try
                {
                    testingGenome.AddConnection(reversed ? testingGenome.VacantConnectionCount - 1 : 0);
                }
                catch (Exception e)
                {
                    throw new InvalidOperationException($"failed to add expected connection [{expectedNewConnection}]", e);
                }

                curConnections.Add(expectedNewConnection);
                expectedVacantCollections.Remove(expectedNewConnection);
                AssertCurrentConnections(testingGenome, curConnections);
                testingGenome.VacantConnectionCount.Should().Be(expectedVacantCollections.Count, "expected vacant connections are: " + string.Join(' ', expectedVacantCollections));
            }
        }

        private static void AssertVacantConnections(Genome genome, ICollection<string> expectedVacantCollections)
        {
            genome.VacantConnectionCount.Should().Be(expectedVacantCollections.Count, "expected vacant connections are: " +
                                                                                      string.Join(' ', expectedVacantCollections));
        }

        private static ConnectionNeurons ConnectionNeuronsFromString(string connectionString)
        {
            var regex = new Regex(@"(\d)(\w)->(\d)(\w)");
            var match = regex.Match(connectionString);
            if (!match.Success)
            {
                throw new InvalidOperationException(
                    $"connection string [{connectionString}] does not match expected mask [{regex}]");
            }

            return new ConnectionNeurons(
                new NeuronGene(GetNeuronType(match.Groups[2].Value), int.Parse(match.Groups[1].Value)),
                new NeuronGene(GetNeuronType(match.Groups[4].Value), int.Parse(match.Groups[3].Value)));

            NeuronGeneType GetNeuronType(string type)
            {
                switch (type)
                {
                    case "h":
                        return NeuronGeneType.Hidden;
                    case "s":
                        return NeuronGeneType.Sensor;
                    case "b":
                        return NeuronGeneType.Bias;
                    case "e":
                        return NeuronGeneType.Effector;
                }

                throw new InvalidOperationException($"unknown neuron type: [{type}]");
            }
        }
    }
}