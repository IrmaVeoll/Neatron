using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Neat;
using Neat.Utils;
using Redzen.Linq;

namespace XorDemo.Model
{
    public sealed class XorNetworkSearch
    {
        private const int ClusterCount = 10;
        
        private static readonly byte[][] XorTruthTable =
        {
            new byte[] {0, 0, 0},
            new byte[] {0, 1, 1},
            new byte[] {1, 0, 1},
            new byte[] {1, 1, 0}
        };

        private readonly PopulationParameters _populationParameters;

        private IEnumerable<Genome> _genomes;
        private readonly Population _neatPopulation;
        private List<ParetoFrontPoint> _archive;
        private readonly int _archiveSize;
        private readonly bool _isRecurrent;

        public XorNetworkSearch(
            PopulationParameters populationParameters,
            NetworkParameters networkParameters,
            ReproductionParameters reproductionParameters)
        {
            _populationParameters = populationParameters;
            _neatPopulation = new Population(networkParameters, reproductionParameters);
            _genomes = Enumerable.Range(0, _populationParameters.PopulationSize)
                .AsParallel()
                .Select(_ => _neatPopulation.CreateInitialGenome()).
                ToList(_populationParameters.PopulationSize);

            _archiveSize = _populationParameters.PopulationSize / 30;
            _archive = new List<ParetoFrontPoint>(_archiveSize);
            _isRecurrent = networkParameters.IsRecurrent;
        }
        
        public SearchResult SearchNext()
        {
            var evaluatedGenomes = _genomes
                .AsParallel()
                .Select(g => (genome: g, fitness: Evaluate(g, _isRecurrent)))
                .ToList(_populationParameters.PopulationSize);
            
            var paretoFrontPointCount = _populationParameters.PopulationSize + _archive.Count;

            var paretoFrontPoints = evaluatedGenomes
                .Select(e => new ParetoFrontPoint(e.genome,
                    e.fitness,
                    _isRecurrent ? 1 : e.genome.NetworkConnectionCount))
                .Concat(_archive)
                .ToList(paretoFrontPointCount);

            //SetSharedFitness(paretoFrontPoints);
            
            var fitnessRating = paretoFrontPoints
                .OrderByDescending(p => p.Fitness)
                .ThenByDescending(p => p.Simplicity)
                .ToList(paretoFrontPointCount);
            var simplicityRating = paretoFrontPoints
                .OrderByDescending(p => p.Simplicity)
                .ThenByDescending(p => p.Fitness)
                .ToList(paretoFrontPointCount);

            SetRanks(fitnessRating, simplicityRating);
            SetSparsity(paretoFrontPoints, 
                fitnessRating.First().Fitness - fitnessRating.Last().Fitness, 
                simplicityRating.First().Simplicity - simplicityRating.Last().Simplicity);

            _archive = paretoFrontPoints
                .OrderBy(p => p.Rank)
                .ThenByDescending(p => p.Sparsity)
                .Take(Math.Min(paretoFrontPoints.Count, _archiveSize))
                .ToList(Math.Max(_archive.Count, _populationParameters.PopulationSize));

            if (!_isRecurrent)
            {
                var samples = SampleBestFitter(fitnessRating[0], 10);

                _ = paretoFrontPoints
                    .AsParallel()
                    .Select(p =>
                    {
                    p.Genome.Network.Train(samples, 0.05f, 0.001f);
                        return p;
                    })
                    .ToList(paretoFrontPoints.Count);
            }

            _genomes = Enumerable.Range(0, _populationParameters.PopulationSize)
                .AsParallel()
                .Select(_ => Reproduce( paretoFrontPoints))
                .ToList(_populationParameters.PopulationSize);
            
            foreach (var paretoFrontPoint in _archive)
            {
                paretoFrontPoint.Rank = -1;
                paretoFrontPoint.Sparsity = 0;
            }

            return new SearchResult(fitnessRating, simplicityRating);
        }

        private static float[][] SampleBestFitter(ParetoFrontPoint best, int sampleCount)
        {
            var sensorCount = best.Genome.Network.Sensors.Count;
            var effectorCount = best.Genome.Network.Effectors.Count;
            var sampleLen = sensorCount + effectorCount;
            
            var samples = new float[sampleCount][];
            
            for (var i = 0; i < sampleCount; i++)
            {
                samples[i] = new float[sampleLen];
                var network = best.Genome.Network;
                for (var s = 0; s < sensorCount; ++s)
                {
                    network.Sensors[s] = RandomSource.Range(-1, 1);
                    samples[i][s] = network.Sensors[s];
                }

                network.Activate();
                
                for (var e = 0; e < effectorCount; ++e)
                {
                    samples[i][sensorCount + e] = network.Effectors[e];
                }
            }
            
            return samples;
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

        private static void SetSharedFitness(IReadOnlyList<ParetoFrontPoint> paretoFrontPoints)
        {
            var centers = EnumerableUtils.RangeRandomOrder(0, paretoFrontPoints.Count, RandomSource.Rng)
                .Take(ClusterCount)
                .Select(i => paretoFrontPoints[i])
                .ToList(ClusterCount);
            
            var disjointSets = Enumerable.Range(0, ClusterCount)
                .Select(_ => new List<ParetoFrontPoint>())
                .ToList(ClusterCount);
            
            for (var j = 0; j < 5; j++)
            {
                foreach (var point in paretoFrontPoints)
                {
                    point.CentroidDistance = float.PositiveInfinity;
                    var index = 0;
                    for (var i = 0; i < ClusterCount; i++)
                    {
                        var dist = point.Genome.GetNeatChromosomesDistance(centers[i].Genome);
                        if (dist < point.CentroidDistance)
                        {
                            point.CentroidDistance = dist;
                            index = i;
                        }
                    }

                    disjointSets[index].Add(point);
                }

                for (var i = 0; i < ClusterCount; i++)
                {
                    if (disjointSets[i].Count > 0)
                    {
                        centers[i] = disjointSets[i].Median();
                    }
                }
            }

            for (var i = 0; i < disjointSets.Count; i++)
            {
                foreach (var point in disjointSets[i])
                {
                    point.SharedFitness /= disjointSets[i].Count;
                }
            }
        }

        private static void SetSparsity(IEnumerable<ParetoFrontPoint> points, float fitnessRange, float simplicityRange)
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

        public static IReadOnlyList<float> EvaluateWinner(ParetoFrontPoint winner, bool isRecurrent)
        {
            Debug.Assert(winner != null, nameof(winner) + " != null");

            var result = new List<float>(XorTruthTable.Length);
            foreach (var t in XorTruthTable)
            {
                result.Add(ActivateAndGetEffector(winner.Genome, isRecurrent, t[0], t[1]));
            }

            return result;
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

        private static float Evaluate(Genome genome, bool isRecurrent)
        {
            if (genome.IsNetworkDisconnected)
            {
                return 0;
            }

            const int iterationCount = 10;
            var maxFitness = (float) XorTruthTable.Length * iterationCount;
            var fitness = maxFitness;


            for (var iteration = 0; iteration < iterationCount; iteration++)
                foreach (var i in EnumerableUtils.RangeRandomOrder(0, XorTruthTable.Length, RandomSource.Rng))
                {
                    var diff = XorTruthTable[i][2] - ActivateAndGetEffector(genome, 
                                   isRecurrent, 
                                   XorTruthTable[i][0], 
                                   XorTruthTable[i][1]);
                    fitness -= diff * diff;
                }

            return fitness / maxFitness;
        }

        private static float ActivateAndGetEffector(Genome genome, bool isRecurrent, byte firstSensor, byte secondSensor)
        {
            if (isRecurrent)
            {
                genome.Network.Sensors[0] = NormalizeInput(firstSensor);
                genome.Network.Activate();
                genome.Network.Sensors[0] = NormalizeInput(secondSensor);
                genome.Network.Activate();
                genome.Network.Activate();
            }
            else
            {
                genome.Network.Sensors[0] = NormalizeInput(firstSensor);
                genome.Network.Sensors[1] = NormalizeInput(secondSensor);
                genome.Network.Activate();
            }

            return NormalizeOutput(genome.Network.Effectors[0]);
        }

        private static float NormalizeInput(float i)
        {
            return i * 2 - 1;
        }
        
        private static float NormalizeOutput(float i)
        {
            return (i + 1) / 2;
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
}