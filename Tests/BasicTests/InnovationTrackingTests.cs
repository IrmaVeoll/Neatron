using Microsoft.Extensions.DependencyModel;
using Neat;
using Xunit;

namespace Tests.BasicTests
{
    public class InnovationTrackingTests
    {
        [Fact]
        public void CheckTrackerSanity()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var first = new Genome(parameters, tracker, false);
            var second = new Genome(parameters, tracker, false);

            // Act
            Assert.Equal(5, first.NeatChromosome.Count);
            Assert.Equal(5, second.NeatChromosome.Count);
            first.SplitConnection(1);
            second.SplitConnection(1);

            // Assert
            Assert.Equal(6, first.NeatChromosome.Count);
            Assert.Equal(6, second.NeatChromosome.Count);
            for (var i = 0; i < first.NeatChromosome.Count; i++)
            {
                var firstGene = first.NeatChromosome[i];
                var secondGene = second.NeatChromosome[i];
                Assert.Equal(firstGene.Id, secondGene.Id);
                Assert.Equal(firstGene.SourceId, secondGene.SourceId);
                Assert.Equal(firstGene.TargetId, secondGene.TargetId);
            }
        }
        
        [Fact]
        public void CheckLinkResurrection()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);

            // Act
            Assert.Equal(0, genome.VacantConnectionCount);
            var oldId = genome.NeatChromosome[1].Id;
            genome.RemoveConnection(1);
            Assert.Equal(1, genome.VacantConnectionCount);
            genome.AddConnection(0);

            // Assert
            Assert.Single(genome.NeatChromosome, gene => gene.Id == oldId);
        }

        [Fact]
        public void CheckLinksAmbiguityHandling()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var first = new Genome(parameters, tracker, false);
            var second = new Genome(parameters, tracker, false);

            // Act
            Assert.True(first.NeatChromosome[1].SourceId == 1 && first.NeatChromosome[1].TargetId == 4);
            Assert.True(second.NeatChromosome[1].SourceId == 1 && second.NeatChromosome[1].TargetId == 4);
            first.SplitConnection(1);
            second.SplitConnection(1);
            while (second.VacantConnectionCount > 0) // make fully connected network to ensure our link is there
                second.AddConnection(0);

            // Assert
            var linkGene = Assert.Single(second.NeatChromosome, gene => gene.SourceId == 1 && gene.TargetId == 4);
            Assert.DoesNotContain(first.NeatChromosome, gene => gene.Id == linkGene.Id);
        }
    }
}