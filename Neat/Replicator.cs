using System;
using System.Collections.Generic;
using Redzen.Numerics.Distributions;
using Redzen.Random;
using static Neat.Utils.RandomSource;
using static Redzen.Numerics.Distributions.DiscreteDistribution;
using static Redzen.Numerics.Distributions.Double.ZigguratGaussian;

namespace Neat
{
    public sealed class Replicator
    {
        private class WeightMutationDistribution
        {
            private readonly IRandomSource _rng;
            private readonly DiscreteDistribution _distribution;
            private readonly IReadOnlyList<WeightMutationInfo> _weightMutations;
            
            internal WeightMutationDistribution(WeightMutations weightMutations, IRandomSource rng)
            {
                _rng = rng;
                _weightMutations = weightMutations.Mutations;
                
                var probArr = new double[_weightMutations.Count];
                var labelArr = new int[_weightMutations.Count];
                for (var i = 0; i < _weightMutations.Count; i++)
                {
                    probArr[i] = _weightMutations[i].RouletteWheelShare;
                    labelArr[i] = i;
                }
                _distribution = new DiscreteDistribution(probArr, labelArr);
            }

            internal WeightMutationInfo Sample() => _weightMutations[DiscreteDistribution.Sample(_rng, _distribution)];
        }

        private enum MutationType
        {
            Weight,
            SplitConnection,
            AddConnection,
            RemoveConnection
        }
        
        private readonly DiscreteDistribution _mutationsDistribution;
        private readonly WeightMutationDistribution _weightMutationDistribution;
        private readonly CrossoverType _crossoverType;

        internal Replicator(ReproductionParameters reproductionParameters)
        {
            _mutationsDistribution = new DiscreteDistribution(new double[]
                {
                    reproductionParameters.WeightMutations.OverallRouletteWheelShare,
                    reproductionParameters.SplitConnectionRouletteWheelShare,
                    reproductionParameters.AddConnectionRouletteWheelShare,
                    reproductionParameters.RemoveConnectionRouletteWheelShare,
                },
                new[]
                {
                    (int) MutationType.Weight,
                    (int) MutationType.SplitConnection,
                    (int) MutationType.AddConnection,
                    (int) MutationType.RemoveConnection
                });
            
            _weightMutationDistribution = new WeightMutationDistribution(reproductionParameters.WeightMutations, Rng);
            _crossoverType = reproductionParameters.CrossoverType;
        }

        public Genome Reproduce(Genome genome)
        {
            var newGenome = genome.Clone();
            Mutate(newGenome);
            return newGenome;
        }

        public Genome Reproduce(Genome betterGenome, Genome worseGenome) =>
            betterGenome.Mate(worseGenome, _crossoverType);

        private void Mutate(Genome genome)
        {
            var mutationsDistribution = _mutationsDistribution;
            while (true)
            {
                var outcome = Sample(Rng, mutationsDistribution);
                bool success;
                switch ((MutationType) outcome)
                {
                    case MutationType.Weight:
                        MutateConnectionWeight(genome, _weightMutationDistribution.Sample());
                        return;
                    case MutationType.SplitConnection:
                        success = TrySplitConnection(genome);
                        break;
                    case MutationType.AddConnection:
                        success = TryAddConnection(genome);
                        break;
                    case MutationType.RemoveConnection:
                        success = TryRemoveConnection(genome);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException($"Unexpected outcome value [{outcome}]");
                }

                if (success)
                {
                    break;
                }
                
                mutationsDistribution = mutationsDistribution.RemoveOutcome(outcome);
                if (mutationsDistribution.Probabilities.Length == 0)
                {
                    return;
                }
            }
        }

        private static bool TrySplitConnection(Genome genome)
        {
            var connectionGenes = genome.NeatChromosome;
            if (connectionGenes.Count == 0)
            {
                return false;
            }
            
            genome.SplitConnection(Next(connectionGenes.Count));
            
            return true;
        }

        private static bool TryAddConnection(Genome genome)
        {
            var vacantConnectionCount = genome.VacantConnectionCount;
            if (vacantConnectionCount == 0)
            {
                return false;
            }

            genome.AddConnection(Next(vacantConnectionCount));

            return true;
        }

        private static bool TryRemoveConnection(Genome genome)
        {
            var connectionGenes = genome.NeatChromosome;
            if (connectionGenes.Count < 2)
            {
                return false;
            }

            genome.RemoveConnection(Next(connectionGenes.Count));

            return true;
        }

        private static void MutateConnectionWeight(Genome genome, WeightMutationInfo sample)
        {
            float GetNewWeight(float oldWeight)
            {
                if (sample is WeightTweak t)
                {
                    return oldWeight + (float)Sample(Rng) * t.Sigma;
                }

                return ConnectionGeneWeightGenerator.GetRandomUniform();
            }

            var connectionGenes = genome.NeatChromosome;

            var mutatingConnectionCount = Math.Min(sample.ConnectionCount, connectionGenes.Count);

            for (var i = 0; i < mutatingConnectionCount; i++)
            {
                var connectionGeneIndex = Next(connectionGenes.Count);
                var connectionGene = connectionGenes[connectionGeneIndex];

                genome.UpdateWeight(in connectionGene, connectionGeneIndex, GetNewWeight(connectionGene.Weight));
            }
        }
    }
}