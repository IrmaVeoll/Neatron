using System;
using System.Collections.Generic;
using System.Linq;
using Neat;
using Neat.Utils;
using static System.Console;
using static System.Linq.Enumerable;
using static System.Math;
using static Neat.ConnectionGene;
using static Neat.CrossoverType;
using static SplitConnectionProblem.Program.SplitConnectionWeightsSearch;

namespace SplitConnectionProblem
{
    class Program
    {
        public readonly struct SearchResult
        {
            public readonly IReadOnlyList<ParetoFrontPoint> FitnessRating;

            public readonly IReadOnlyList<ParetoFrontPoint> SimplicityRating;

            internal SearchResult(
                IReadOnlyList<ParetoFrontPoint> fitnessRating,
                IReadOnlyList<ParetoFrontPoint> simplicityRating)
            {
                FitnessRating = fitnessRating;
                SimplicityRating = simplicityRating;
            }
        }

        public readonly struct PopulationParameters
        {
            public readonly int PopulationSize;
            public readonly int TournamentSize;

            public PopulationParameters(int populationSize, int tournamentSize)
            {
                PopulationSize = populationSize;
                TournamentSize = tournamentSize;
            }
        }

        public class ParetoFrontPoint
        {
            internal ParetoFrontPoint(Genome genome, float fitness, int complexity)
            {
                Genome = genome;
                Fitness = fitness;
                Complexity = complexity;
                Simplicity = -complexity; //complexity == 0 ? 1f : (1f / complexity);
            }

            public Genome Genome { get; }
            public float Fitness { get; }
            public float Simplicity { get; }
            public int Complexity { get; }
            public int Rank { get; set; } = -1;
            public float Sparsity { get; set; }

            public bool IsBetterThan(ParetoFrontPoint other)
            {
                if (this == other) return false;
                
                if (Rank < other.Rank) return true;
                if (Rank > other.Rank) return false;

                if (Sparsity < other.Sparsity) return false;
                if (Sparsity > other.Sparsity) return true;

                if (Fitness > other.Fitness) return true;
                if (Fitness < other.Fitness) return false;

                if (Simplicity > other.Simplicity) return true;
                if (Simplicity < other.Simplicity) return false;

                return false;
            }

            public bool IsDominatedBy(ParetoFrontPoint other)
            {
                return other.Fitness >= Fitness &&
                       other.Simplicity > Simplicity ||
                       other.Fitness > Fitness &&
                       other.Simplicity >= Simplicity;
            }

            public override string ToString() =>
                $"Fitness: {Fitness:0.0000} Simplicity: {Simplicity:0.00} Rank: {Rank} Sparsity: {Sparsity:0.0000} Complexity: {Complexity}";
        }

        public sealed class SplitConnectionWeightsSearch
        {
            private readonly PopulationParameters _populationParameters;

            private IEnumerable<Genome> _genomes;
            private readonly Population _neatPopulation;
            private List<ParetoFrontPoint> _archive;
            private readonly int _archiveSize;

            public SplitConnectionWeightsSearch(
                PopulationParameters populationParameters,
                NetworkParameters networkParameters,
                ReproductionParameters reproductionParameters)
            {
                _populationParameters = populationParameters;
                _neatPopulation = new Population(networkParameters, reproductionParameters, false);
                _genomes = Range(0, _populationParameters.PopulationSize)
                    .Select(_ => _neatPopulation.CreateInitialGenome());

                _archiveSize = _populationParameters.PopulationSize / 2;
                _archive = new List<ParetoFrontPoint>(_archiveSize);
                
                WeightSamples = Range(0, 300)
                    .Select(_ => RandomSource.Range(0, MaxAbsWeightValue))
                    .ToArray();
                
                InputSamples = Range(0, 60)
                    .Select(_ => RandomSource.Range(0, 1))
                    .ToArray();
            }

            public static float[] WeightSamples { get; private set; }

            public static float[] InputSamples { get; private set; }

            public SearchResult SearchNext()
            {
                var evaluatedGenomes = _genomes
                    .AsParallel()
                    .Select(g => (genome: g, fitness: Evaluate(g)))
                    .ToList();

                var paretoFrontPoints = evaluatedGenomes
                    .Select(e => new ParetoFrontPoint(e.genome,
                        e.fitness,
                        e.genome.NetworkConnectionCount))
                    .Concat(_archive)
                    .ToList();

                var fitnessRating = paretoFrontPoints
                    .OrderByDescending(p => p.Fitness)
                    .ThenByDescending(p => p.Simplicity)
                    .ToList();
                var simplicityRating = paretoFrontPoints
                    .OrderByDescending(p => p.Simplicity)
                    .ThenByDescending(p => p.Fitness)
                    .ToList();

                SetRanks(fitnessRating, simplicityRating);
                SetSparsity(paretoFrontPoints,
                    fitnessRating.First().Fitness - fitnessRating.Last().Fitness,
                    simplicityRating.First().Simplicity - simplicityRating.Last().Simplicity);

                _archive = paretoFrontPoints
                    .OrderBy(p => p.Rank)
                    .ThenByDescending(p => p.Sparsity)
                    .Take(Min(paretoFrontPoints.Count, _archiveSize))
                    .ToList();

                _genomes = Range(0, _populationParameters.PopulationSize)
                    .AsParallel()
                    .Select(_ => Reproduce(paretoFrontPoints))
                    .ToList();

                foreach (var paretoFrontPoint in _archive)
                {
                    paretoFrontPoint.Rank = -1;
                    paretoFrontPoint.Sparsity = 0;
                }

                return new SearchResult(fitnessRating, simplicityRating);
            }

            private Genome Reproduce(IReadOnlyList<ParetoFrontPoint> paretoFrontPoints)
            {
                var p1 = MakeTournamentSelection(paretoFrontPoints);
                if (RandomSource.Next() < 0.8f)
                {
                    return _neatPopulation.Replicator.Reproduce(p1.Genome);
                }

                var p2 = MakeTournamentSelection(paretoFrontPoints);

                return p1.Fitness > p2.Fitness
                    ? _neatPopulation.Replicator.Reproduce(p1.Genome, p2.Genome)
                    : _neatPopulation.Replicator.Reproduce(p2.Genome, p1.Genome);
            }

            private static void SetSparsity(IEnumerable<ParetoFrontPoint> points, float fitnessRange,
                float simplicityRange)
            {
                void AssignSparsity(IEnumerable<ParetoFrontPoint> front,
                    Func<ParetoFrontPoint, float> objectiveSelector,
                    float objectiveRange)
                {
                    var objectiveRating = front.OrderBy(objectiveSelector).ToList();

                    objectiveRating[0].Sparsity = float.PositiveInfinity;
                    objectiveRating[objectiveRating.Count - 1].Sparsity =
                        objectiveRange > float.Epsilon ? float.PositiveInfinity : 0;

                    for (var i = 1; i < objectiveRating.Count - 1; i++)
                    {
                        objectiveRating[i].Sparsity += objectiveRange > float.Epsilon
                            ? (objectiveSelector(objectiveRating[i + 1]) - objectiveSelector(objectiveRating[i])) /
                              objectiveRange
                            : 0;
                    }
                }

                var fronts = points.GroupBy(p => p.Rank);
                foreach (var front in fronts)
                {
                    AssignSparsity(front, point => point.Fitness, fitnessRange);
                    AssignSparsity(front, point => point.Simplicity, simplicityRange);
                }
            }

            private ParetoFrontPoint MakeTournamentSelection(IReadOnlyList<ParetoFrontPoint> paretoFrontItems)
            {
                var best = paretoFrontItems[RandomSource.Next(paretoFrontItems.Count)];
                for (var i = 2; i <= _populationParameters.TournamentSize; i++)
                {
                    var next = paretoFrontItems[RandomSource.Next(paretoFrontItems.Count)];
                    if (next.IsBetterThan(best)) best = next;
                }

                return best;
            }

            private static void SetRanks(IReadOnlyList<ParetoFrontPoint> fitnessRating,
                IReadOnlyList<ParetoFrontPoint> complexityRating)
            {
                var paretoRanks = new List<ParetoFrontPoint>(fitnessRating.Count);
                for (var i = 0; i < fitnessRating.Count; i++)
                {
                    if (fitnessRating[i].Rank == -1)
                        SetParetoRank(fitnessRating[i], paretoRanks);

                    if (complexityRating[i].Rank == -1)
                        SetParetoRank(complexityRating[i], paretoRanks);
                }
            }

            public static float ZeroCenteredSoftSign(float input)
            {
                var inputAbs = MathF.Abs(input);
                return inputAbs > float.Epsilon ? input / (0.5f + inputAbs) : 0;
            }

            private static float GetNetworkError(Genome genome)
            {
                if (genome.IsNetworkDisconnected)
                {
                    return float.MaxValue;
                }
                
                float errSum = 0;
                foreach (var w in WeightSamples)
                {
                    foreach (var x in InputSamples)
                    {
                        genome.Network.Sensors[0] = w / MaxAbsWeightValue * 2 - 1;
                        genome.Network.Activate();

                        var w1 = (genome.Network.Effectors[0] + 1) / 2 * MaxAbsWeightValue;
                        var w2 = (genome.Network.Effectors[1] + 1) / 2 * MaxAbsWeightValue;

                        var predicted = ZeroCenteredSoftSign(w1 * x) * w2;
                        var err = predicted - w * x;
                        errSum += err * err;
                    }
                }

                return errSum/(WeightSamples.Length * InputSamples.Length);
            }

            public static float Evaluate(Genome genome)
            {
                var err = GetNetworkError(genome); 
                
                if (!(err > float.Epsilon))
                {
                    return float.MaxValue;
                }

                return -err;
            }

            private static void SetParetoRank(ParetoFrontPoint point, ICollection<ParetoFrontPoint> ranks)
            {
                var maxNonDominantFrontNumber = 0;
                foreach (var r in ranks)
                    if (point.IsDominatedBy(r) && maxNonDominantFrontNumber < r.Rank)
                        maxNonDominantFrontNumber = r.Rank;

                point.Rank = maxNonDominantFrontNumber + 1;
                ranks.Add(point);
            }
        }

        private static readonly PopulationParameters PopulationParams =
            new PopulationParameters(150, 2);

        private static readonly NetworkParameters NetworkParameters =
            new NetworkParameters(1, 2, NetworkType.FeedForward)
            {
                InitialConnectionDensity = 0.9f
            };

        private static readonly ReproductionParameters ReproductionParameters =
            new ReproductionParameters
            {
                CrossoverType = ArithmeticRecombination,

                WeightMutations = new WeightMutations
                {
                    OverallRouletteWheelShare = 110,
                    Mutations =
                    {
                        new WeightTweak
                        {
                            RouletteWheelShare = 100f,
                            ConnectionCount = 1,
                            Sigma = 0.1f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 20f,
                            ConnectionCount = 2,
                            Sigma = 0.05f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 20f,
                            ConnectionCount = 3,
                            Sigma = 0.025f
                        },
                        new WeightPerturb
                        {
                            RouletteWheelShare = 10f,
                            ConnectionCount = 1
                        },
                    }
                },
                AddConnectionRouletteWheelShare = 5f,
                RemoveConnectionRouletteWheelShare = 5f,
                SplitConnectionRouletteWheelShare = 2f
            };

        private static (float, float) GetWeightsFromApproxModel(float w)
        {
            const float a = 0.0000127212307424923f;
            const float b = 0.000187725313497866f;
            const float c = 0.00173497083092954f;
            const float d = 0.015099962668533f;
            const float e = 0.0999436144686452f;

            var wSquared = w * w;

            var w1 = a * wSquared * wSquared * w + b * wSquared * wSquared + c * wSquared * w + d * wSquared + e * w;

            return (w1, MaxAbsWeightValue * MathF.Sign(w));
        }

        private static float GetModelError(IReadOnlyCollection<float> weightSamples, IReadOnlyCollection<float> inputSamples)
        {
            var errSum = 0f;
            foreach (var w in weightSamples)
            {
                foreach (var x in inputSamples)
                {
                    var (w1, w2) = GetWeightsFromApproxModel(w);

                    var predicted = ZeroCenteredSoftSign(w1 * x) * w2;
                    var err = predicted - w * x;
                    errSum += err * err;
                }
            }

            return errSum/(weightSamples.Count * inputSamples.Count);
        }

        static void Main(string[] args)
        {
            SearchResult searchResult = default;

            while (true)
            {
                var searchModel = new SplitConnectionWeightsSearch(
                    PopulationParams,
                    NetworkParameters,
                    ReproductionParameters);

                for (var generation = 0; generation < 500; ++generation)
                {   
                    var newSearchResult = searchModel.SearchNext();
                    searchResult = newSearchResult;
                    
                    if (generation == 0)
                    {
                        WriteLine($"Initial network MSE: {Abs(Evaluate(searchResult.FitnessRating.First().Genome))}");
                    }
                }

                var genome = searchResult.FitnessRating.First().Genome;
            
                var networkError = Abs(Evaluate(genome));
                var modelError = GetModelError(WeightSamples, InputSamples);
            
                WriteLine($"Model MSE: {modelError}");
                WriteLine($"Evolved network MSE: {networkError}");
                
                for (var i = 0; i <= MaxAbsWeightValue; i++)
                {
                    genome.Network.Sensors[0] = i / MaxAbsWeightValue * 2f - 1f;
                    genome.Network.Activate();
                
                    WriteLine($"w = {i}, w1 = {(genome.Network.Effectors[0] + 1) / 2 * MaxAbsWeightValue}, w2 = {(genome.Network.Effectors[1] + 1) / 2 * MaxAbsWeightValue}");
                }

                if (networkError <= modelError)
                {
                    break;
                }
            }
        }
    }
}