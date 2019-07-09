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
            // Arrange
            var parameters = new NetworkParameters(3, 1, NetworkType.Recurrent) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);

            // Act
            genome.SplitConnection(0); // 0 -> 4 into 0 -> 5 -> 4
            genome.SplitConnection(5); // 0 -> 5 -> 4 into 0 -> 6 -> 5 -> 4
            genome.SplitConnection(2); // 2 -> 4

            // Assert
            Assert.Equal(24, genome.VacantConnectionCount);

            // Act
            for (var i = 24; i > 0; i--)
            {
                genome.AddConnection(RandomSource.Next(i));
                Assert.Equal(i - 1, genome.VacantConnectionCount);
            }
        }
        
        [Fact]
        public void AddConnectionsOneByOneFeedforward()
        {
            // Arrange
            var parameters = new NetworkParameters(3, 1, NetworkType.FeedForward) {InitialConnectionDensity = 1f};
            var tracker =
                new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

            var genome = new Genome(parameters, tracker, false);
            
            // Act
            genome.SplitConnection(0); // 0 -> 4 into 0 -> 5 -> 4
            genome.SplitConnection(4); // 0 -> 5 -> 4 into 0 -> 6 -> 5 -> 4
            genome.SplitConnection(1); // 2 -> 4
            
            // Assert
            Assert.Equal(18, genome.VacantConnectionCount);
            
            // Act
            for (var i = genome.VacantConnectionCount; i > 0; i--)
            {
                genome.AddConnection(RandomSource.Next(i));
                Assert.Equal(i-1, genome.VacantConnectionCount);
            }
        }
    }
}