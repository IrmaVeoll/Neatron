using System.Linq;
using Neat;
using Tests.Fakes;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]
namespace Tests.BasicTests
{
    public class Tests
    {
        [Fact]
        public void CheckRandomSourceDrillIn()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                
                // Act
                source.GetParanoid();
                source.PushNextFloats(1, 2, 3);

                // Assert
                Assert.Equal(1, Neat.Utils.RandomSource.Next());
                Assert.Equal(2, Neat.Utils.RandomSource.Next());
                Assert.Equal(3, Neat.Utils.RandomSource.Next());
            }
        }
        
        [Fact]
        public void CheckArithmeticRecombination()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange

                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 2); // crossover points
                source.PushNextBools(true); // start from best
                var result = best.Mate(worst, CrossoverType.ArithmeticRecombination);

                // Assert
                for (var i = 0; i < result.NeatChromosome.Count; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight,
                                (best.NeatChromosome[i].Weight + worst.NeatChromosome[i].Weight) / 2f);
                }
            }
        }
        
        [Fact]
        public void CheckUniformRecombination()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange

                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 2); // crossover points
                source.PushNextBools(true); // start from best
                source.PushNextBools(false, true, false, true, false); // best, worst, best, worst, best
                var result = best.Mate(worst, CrossoverType.Uniform);

                // Assert
                for (var i = 0; i < result.NeatChromosome.Count; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight, i % 2 == 0
                        ? best.NeatChromosome[i].Weight
                        : worst.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckOnePointRecombinationFromBest()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 3); // crossover points
                source.PushNextBools(true); // start from best
                var result = best.Mate(worst, CrossoverType.OnePoint);

                // Assert
                Assert.Equal(result.NeatChromosome[0].Weight, best.NeatChromosome[0].Weight);
                for (var i = 1; i < result.NeatChromosome.Count; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight, worst.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckOnePointRecombinationFromWorst()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 3); // crossover points
                source.PushNextBools(false); // start from best
                var result = best.Mate(worst, CrossoverType.OnePoint);

                // Assert
                Assert.Equal(result.NeatChromosome[0].Weight, worst.NeatChromosome[0].Weight);
                for (var i = 1; i < result.NeatChromosome.Count; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight, best.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckTwoPointRecombinationFromBest()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 3); // crossover points
                source.PushNextBools(true); // start from best
                var result = best.Mate(worst, CrossoverType.TwoPoints);

                // Assert
                Assert.Equal(result.NeatChromosome[0].Weight, best.NeatChromosome[0].Weight);
                Assert.Equal(result.NeatChromosome[4].Weight, best.NeatChromosome[4].Weight);
                for (var i = 1; i < 4; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight, worst.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckTwoPointRecombinationFromWorst()
        {
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(1, 3); // crossover points
                source.PushNextBools(false); // start from best
                var result = best.Mate(worst, CrossoverType.TwoPoints);

                // Assert
                Assert.Equal(result.NeatChromosome[0].Weight, worst.NeatChromosome[0].Weight);
                Assert.Equal(result.NeatChromosome[4].Weight, worst.NeatChromosome[4].Weight);
                for (var i = 1; i < 4; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Weight, best.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckTwoPointRecombinationOfSingleGene()
        {
            // TODO fix crossover, make this one green
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(2, 2); // crossover points
                source.PushNextBools(true); // start from best
                var result = best.Mate(worst, CrossoverType.TwoPoints);

                // Assert
                Assert.Equal(result.NeatChromosome[2].Weight, worst.NeatChromosome[2].Weight);
                for (var i = 0; i < result.NeatChromosome.Count; i++)
                {
                    if (i == 2)
                        continue;
                    Assert.Equal(result.NeatChromosome[i].Weight, best.NeatChromosome[i].Weight);
                }
            }
        }
        
        [Fact]
        public void CheckDisjointsAndEtcAreTakenFromBest()
        {
            // TODO fix crossover, make this one green
            using (var source = RandomSource.DrillIn())
            {
                // Arrange
                var parameters = new NetworkParameters(3, 1) {InitialConnectionDensity = 1f};
                var tracker =
                    new InnovationTracker(NetworkParameters.BiasCount + parameters.SensorCount + parameters.EffectorCount);

                var best = new Genome(parameters, tracker, true);
                var worst = new Genome(parameters, tracker, true);
                best.SplitConnection(1);

                // Act
                source.GetParanoid();
                source.PushLimitedNexts(0, 2); // crossover points - all from worst
                source.PushNextBools(true); // start from best
                var result = best.Mate(worst, CrossoverType.OnePoint);

                // Assert
                Assert.Equal(best.NeatChromosome.Count, result.NeatChromosome.Count);
                for (var i = 0; i < result.NeatChromosome.Count - 2; i++)
                {
                    var parentGene = Assert.Single(worst.NeatChromosome, g => g.Id == result.NeatChromosome[i].Id);
                    Assert.Equal(result.NeatChromosome[i].Weight, parentGene.Weight);
                }
                for (var i = result.NeatChromosome.Count - 2; i < result.NeatChromosome.Count; i++)
                {
                    Assert.Equal(result.NeatChromosome[i].Id, best.NeatChromosome[i].Id);
                    Assert.Equal(result.NeatChromosome[i].Weight, best.NeatChromosome[i].Weight);
                }
            }
        }
    }
}